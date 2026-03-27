from collections import defaultdict
from collections.abc import ValuesView
from contextlib import ExitStack
import fcntl
import glob
import json
from logging import getLogger
import os
from pathlib import Path
import pickle
from typing import BinaryIO
import warnings

from .metrics import Gauge
from .metrics_core import Metric
from .mmap_dict import MmapedDict
from .samples import Sample
from .utils import floatToGoString

try:  # Python3
    FileNotFoundError
except NameError:  # Python >= 2.5
    FileNotFoundError = IOError

log = getLogger(__name__)


def reduce_metrics(*metrics: dict[str, Metric]) -> dict[str, Metric]:
    """Merge multiple dicts of metrics into a single dict, by extending the samples of metrics with the same name."""
    result = {}
    for metric_dict in metrics:
        for name, metric in metric_dict.items():
            try:
                result[name].samples.extend(metric.samples)
            except KeyError:  # noqa: PERF203
                result[name] = metric
    return result


class MultiProcessCollector:
    """Collector for files for multi-process mode."""

    def __init__(self, registry, path=None):
        if path is None:
            # This deprecation warning can go away in a few releases when removing the compatibility
            if 'prometheus_multiproc_dir' in os.environ and 'PROMETHEUS_MULTIPROC_DIR' not in os.environ:
                os.environ['PROMETHEUS_MULTIPROC_DIR'] = os.environ['prometheus_multiproc_dir']
                warnings.warn("prometheus_multiproc_dir variable has been deprecated in favor of the upper case naming PROMETHEUS_MULTIPROC_DIR", DeprecationWarning)
            path = os.environ.get('PROMETHEUS_MULTIPROC_DIR')
        if not path or not os.path.isdir(path):
            raise ValueError('env PROMETHEUS_MULTIPROC_DIR is not set or not a directory')
        self._path = path
        if registry:
            registry.register(self)

    @staticmethod
    def merge(files, accumulate=True) -> ValuesView[Metric]:
        """Merge metrics from given mmap files.

        By default, histograms are accumulated, as per prometheus wire format.
        But if writing the merged data back to mmap files, use
        accumulate=False to avoid compound accumulation.
        """
        metrics = MultiProcessCollector._read_metrics(files)
        return MultiProcessCollector._accumulate_metrics(metrics, accumulate)

    @staticmethod
    def _read_metrics(files) -> dict[str, Metric]:
        # read all .db files and collect all samples for same metric together
        metrics = {}
        key_cache = {}

        def _parse_key(key):
            val = key_cache.get(key)
            if not val:
                metric_name, name, labels, help_text = json.loads(key)
                labels_key = tuple(sorted(labels.items()))
                val = key_cache[key] = (metric_name, name, labels, labels_key, help_text)
            return val

        for f in files:
            parts = os.path.basename(f).split('_')
            typ = parts[0]
            try:
                file_values = MmapedDict.read_all_values_from_file(f)
            except FileNotFoundError:
                if typ == 'gauge' and parts[1].startswith('live'):
                    # Files for 'live*' gauges can be deleted between the glob of collect
                    # and now (via a mark_process_dead call) so don't fail if
                    # the file is missing
                    continue
                raise
            for key, value, timestamp, _ in file_values:
                metric_name, name, labels, labels_key, help_text = _parse_key(key)

                metric = metrics.get(metric_name)
                if metric is None:
                    metric = Metric(metric_name, help_text, typ)
                    metrics[metric_name] = metric

                if typ == 'gauge':
                    pid = parts[2][:-3]
                    metric._multiprocess_mode = parts[1]
                    metric.add_sample(name, labels_key + (('pid', pid),), value, timestamp)
                else:
                    # The duplicates and labels are fixed in the next for.
                    metric.add_sample(name, labels_key, value)
        return metrics

    @staticmethod
    def _accumulate_metrics(metrics: dict[str, Metric], accumulate: bool) -> ValuesView[Metric]:
        # refactor (accumulate) samples in each metrics object
        for metric in metrics.values():
            samples = defaultdict(lambda: defaultdict(float))
            sample_timestamps = defaultdict(lambda: defaultdict(float))
            buckets = defaultdict(lambda: defaultdict(float))
            for s in metric.samples:
                name, labels, value, timestamp, exemplar, native_histogram_value = s

                if (
                    metric.type == 'gauge'
                    and metric._multiprocess_mode in (
                        'min', 'livemin',
                        'max', 'livemax',
                        'sum', 'livesum',
                        'mostrecent', 'livemostrecent',
                    )
                ):
                    labels = tuple(l for l in labels if l[0] != 'pid')

                if metric.type == 'gauge':
                    if metric._multiprocess_mode in ('min', 'livemin'):
                        current = samples[labels].setdefault((name, labels), value)
                        if value < current:
                            samples[labels][(name, labels)] = value
                    elif metric._multiprocess_mode in ('max', 'livemax'):
                        current = samples[labels].setdefault((name, labels), value)
                        if value > current:
                            samples[labels][(name, labels)] = value
                    elif metric._multiprocess_mode in ('sum', 'livesum'):
                        samples[labels][(name, labels)] += value
                    elif metric._multiprocess_mode in ('mostrecent', 'livemostrecent'):
                        current_timestamp = sample_timestamps[labels][name]
                        timestamp = float(timestamp or 0)
                        if current_timestamp < timestamp:
                            samples[labels][(name, labels)] = value
                            sample_timestamps[labels][name] = timestamp
                    else:  # all/liveall
                        samples[labels][(name, labels)] = value

                elif metric.type == 'histogram':
                    # A for loop with early exit is faster than a genexpr
                    # or a listcomp that ends up building unnecessary things
                    for l in labels:
                        if l[0] == 'le':
                            bucket_value = float(l[1])
                            # _bucket
                            without_le = tuple(l for l in labels if l[0] != 'le')
                            buckets[without_le][bucket_value] += value
                            break
                    else:  # did not find the `le` key
                        # _sum/_count
                        samples[labels][(name, labels)] += value
                else:
                    # Counter and Summary.
                    samples[labels][(name, labels)] += value

            # Accumulate bucket values.
            if metric.type == 'histogram':
                for labels, values in buckets.items():
                    acc = 0.0
                    for bucket, value in sorted(values.items()):
                        sample_key = (
                            metric.name + '_bucket',
                            labels + (('le', floatToGoString(bucket)),),
                        )
                        if accumulate:
                            acc += value
                            samples[labels][sample_key] = acc
                        else:
                            samples[labels][sample_key] = value
                    if accumulate:
                        samples[labels][(metric.name + '_count', labels)] = acc

            # Convert to correct sample format.
            metric.samples = []
            for _, samples_by_labels in samples.items():
                for (name_, labels), value in samples_by_labels.items():
                    metric.samples.append(Sample(name_, dict(labels), value))
        return metrics.values()

    def collect(self):
        files = glob.glob(os.path.join(self._path, '*.db'))
        return self.merge(files, accumulate=True)


class FlockMultiProcessCollector(MultiProcessCollector):

    MERGED_METRICS_FILENAME = "merged_metrics.pkl"

    def collect(self, recursively: bool = True) -> ValuesView[Metric]:
        """
        Collect metrics from all .db files, merge them with existing merged metrics and return the result.
        """
        folder = Path(self._path)
        current_metrics: dict[str, Metric] = self._read_metrics(str(file) for file in folder.glob('**/*.db' if recursively else '*.db'))
        merged_metrics: list[dict[str, Metric]] = [
            pickle.loads(data)
            for file in folder.glob(f"**/{self.MERGED_METRICS_FILENAME}" if recursively else self.MERGED_METRICS_FILENAME)
            if (data := file.read_bytes())
        ]

        reduced_metrics = reduce_metrics(current_metrics, *merged_metrics)
        return self._accumulate_metrics(reduced_metrics, accumulate=True)

    @classmethod
    def cleanup(cls, folder: Path) -> None:
        """
        Collect all stale `.db` files and merge them into single merged metrics file.
        """

        with ExitStack() as exit_stack:
            merged_file_path = folder / cls.MERGED_METRICS_FILENAME
            merged_file_path.touch(exist_ok=True)
            merged_file = merged_file_path.open("r+b")
            exit_stack.enter_context(merged_file)

            try:
                fcntl.flock(merged_file, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                log.debug("Could not acquire lock on merged metrics file, skipping cleanup")
                return

            files_to_merge: list[BinaryIO] = []
            for file_path in folder.glob('*.db'):
                file = file_path.open('rb')
                try:
                    fcntl.flock(file, fcntl.LOCK_EX | fcntl.LOCK_NB)
                except BlockingIOError:
                    log.debug("Could not acquire lock on file %s, skipping it", file_path)
                    file.close()
                    continue
                exit_stack.enter_context(file)
                files_to_merge.append(file)

            if not files_to_merge:
                return

            # read all .db files and collect all samples for same metric together
            current_metrics: dict[str, Metric] = cls._read_metrics(file.name for file in files_to_merge)

            # load existing merged metrics, if any
            merged_data = merged_file.read()
            try:
                merged_metrics = pickle.loads(merged_data) if merged_data else {}
            except (pickle.PickleError, EOFError):
                merged_metrics = {}

            # extend existing merged metrics with current ones
            reduced_metrics = reduce_metrics(merged_metrics, current_metrics)

            merged_file.seek(0)
            pickle.dump(reduced_metrics, merged_file)
            merged_file.truncate()

            for file in files_to_merge:
                Path(file.name).unlink()


_LIVE_GAUGE_MULTIPROCESS_MODES = {m for m in Gauge._MULTIPROC_MODES if m.startswith('live')}


def mark_process_dead(pid, path=None):
    """Do bookkeeping for when one process dies in a multi-process setup."""
    if path is None:
        path = os.environ.get('PROMETHEUS_MULTIPROC_DIR', os.environ.get('prometheus_multiproc_dir'))
    for mode in _LIVE_GAUGE_MULTIPROCESS_MODES:
        for f in glob.glob(os.path.join(path, f'gauge_{mode}_{pid}.db')):
            os.remove(f)

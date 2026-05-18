#!/usr/bin/env python3
"""
Custom factor calculator using AlphaAgent expression parser.
Supports the same expression syntax as factor mining.

Features:
1. Parse factor expressions (expr_parser)
2. Compute factor values (function_lib)
3. Output Qlib DataLoader-compatible format
4. Load precomputed factors from cache
"""

import hashlib
import json
import logging
import os
import sys
import tempfile
import warnings
from dataclasses import asdict, dataclass, field
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import pandas as pd

warnings.filterwarnings('ignore', category=FutureWarning, module='pandas')
warnings.filterwarnings('ignore', category=UserWarning, module='qlib.contrib.quantaalpha_compat')

# Use thread backend for joblib to avoid subprocess importing LLM modules
os.environ.setdefault('JOBLIB_START_METHOD', 'loky')

logger = logging.getLogger(__name__)

DEFAULT_CACHE_DIR = Path(
    os.environ.get(
        "FACTOR_CACHE_DIR",
        str(Path(__file__).resolve().parents[3] / "outputs/factor_cache"),
    )
)
_WORKER_DATA_DF: Optional[pd.DataFrame] = None


@dataclass
class FactorBatchDiagnostics:
    total_factors: int
    success_count: int = 0
    fail_count: int = 0
    cache_hit_count: int = 0
    cache_location_hit_count: int = 0
    compute_count: int = 0
    skipped_count: int = 0
    aligned_result_count: int = 0
    factor_sources: Dict[str, str] = field(default_factory=dict)
    failed_factors: List[str] = field(default_factory=list)
    missing_cache_locations: List[Dict[str, str]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _init_factor_compute_worker(data_path: str):
    global _WORKER_DATA_DF
    _WORKER_DATA_DF = pd.read_pickle(data_path)


def _compute_factor_worker(task: tuple[str, str]) -> tuple[str, str, Optional[pd.Series], Optional[str]]:
    factor_name, factor_expr = task
    try:
        calculator = CustomFactorCalculator(
            data_df=_WORKER_DATA_DF,
            cache_dir=DEFAULT_CACHE_DIR,
            auto_extract_cache=False,
            config=None,
        )
        result = calculator.calculate_factor(factor_name, factor_expr)
        return factor_name, factor_expr, result, None
    except Exception as e:
        return factor_name, factor_expr, None, str(e)


class CustomFactorCalculator:
    """
    Custom factor calculator using AlphaAgent expression parser and function lib.
    Loads precomputed factors from cache; can auto-extract cache from main program logs.
    """
    
    def __init__(self, data_df: Optional[pd.DataFrame] = None, cache_dir: Optional[Path] = None, 
                 auto_extract_cache: bool = True, config: Optional[Dict] = None):
        """
        Args:
            data_df: Stock data DataFrame (optional, lazy-loaded).
            cache_dir: Cache directory path (optional).
            auto_extract_cache: Whether to auto-extract cache from main program logs (default True).
            config: Config dict for lazy loading data (optional).
        """
        self._raw_data_df = data_df
        self._data_prepared = False
        self._config = config
        self.cache_dir = cache_dir or DEFAULT_CACHE_DIR
        self.auto_extract_cache = auto_extract_cache
        self._cache_extracted = False
        
        if data_df is not None and len(data_df) > 0:
            self._prepare_data()
    
    @property
    def data_df(self) -> pd.DataFrame:
        """Lazy-load stock data."""
        if not self._data_prepared:
            if self._raw_data_df is None or len(self._raw_data_df) == 0:
                if self._config is not None:
                    print("  Loading stock data (needed for expression-based factor computation)...")
                    self._raw_data_df = get_qlib_stock_data(self._config)
                else:
                    raise ValueError("No stock data provided and no config for loading")
            self._prepare_data()
        return self._raw_data_df
        
    def _prepare_data(self):
        """Prepare data and add common derived columns."""
        if self._data_prepared:
            return
        
        df = self._raw_data_df.copy()
        
        if '$return' not in df.columns:
            df['$return'] = df.groupby('instrument')['$close'].transform(
                lambda x: x / x.shift(1) - 1
            )
        
        if df.index.duplicated().any():
            dup_count = df.index.duplicated().sum()
            logger.warning(f"Data has {dup_count} duplicate index entries, deduplicated")
            df = df[~df.index.duplicated(keep='last')]
        
        self._raw_data_df = df
        self._data_prepared = True
        logger.debug(f"Data prepared: {len(df)} rows, cols: {list(df.columns)}")
    
    def _get_cache_key(self, expr: str) -> str:
        """Cache key from expression MD5 hash."""
        return hashlib.md5(expr.encode()).hexdigest()
    
    def _load_from_cache(self, expr: str) -> Optional[pd.Series]:
        """Load factor values from cache."""
        cache_key = self._get_cache_key(expr)
        cache_file = self.cache_dir / f"{cache_key}.pkl"
        
        if cache_file.exists():
            try:
                result = pd.read_pickle(cache_file)
                return self._process_cached_result(result, cache_key)
            except Exception as e:
                logger.debug(f"Cache load failed [{cache_key}]: {e}")
                return None
        return None
    
    def _load_from_cache_location(self, cache_location: Dict) -> Optional[pd.Series]:
        """Load factor from path given in cache_location."""
        if not cache_location:
            return None
        
        result_h5_path = cache_location.get('result_h5_path', '')
        if not result_h5_path:
            return None
        
        h5_file = Path(result_h5_path)
        if not h5_file.exists():
            logger.debug(f"Cache file not found: {result_h5_path}")
            return None
        
        try:
            result = pd.read_hdf(str(h5_file))
            return self._process_cached_result(result, result_h5_path)
        except Exception as e:
            logger.debug(f"Load from cache_location failed [{result_h5_path}]: {e}")
            return None
    
    def _process_cached_result(self, result: Any, source: str) -> Optional[pd.Series]:
        """Normalize cached result format (does not touch self.data_df to avoid lazy load)."""
        try:
            if isinstance(result, pd.DataFrame):
                if len(result.columns) == 1:
                    result = result.iloc[:, 0]
                elif 'factor' in result.columns:
                    result = result['factor']
                else:
                    result = result.iloc[:, 0]
            
            # Standard order: (datetime, instrument)
            if isinstance(result.index, pd.MultiIndex):
                cache_idx_names = list(result.index.names)
                expected_order = ['datetime', 'instrument']
                if cache_idx_names != expected_order and set(cache_idx_names) == set(expected_order):
                    result = result.swaplevel()
                    result = result.sort_index()
            
            return result
        except Exception as e:
            logger.debug(f"Process cached result failed [{source}]: {e}")
            return None
    
    def _save_to_cache(self, expr: str, result: pd.Series):
        """Save factor values to cache."""
        try:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            cache_key = self._get_cache_key(expr)
            cache_file = self.cache_dir / f"{cache_key}.pkl"
            result.to_pickle(cache_file)
        except Exception as e:
            logger.warning(f"Save to cache failed: {e}")
    
    def _auto_extract_cache_from_logs(self):
        """Auto-extract cache from main program logs; runs once on first need."""
        if self._cache_extracted:
            return
        
        self._cache_extracted = True
        logger.debug("Cache extractor is not vendored into qlib; skip auto-extract")
        
    def calculate_factor(self, factor_name: str, factor_expression: str) -> Optional[pd.Series]:
        """
        Compute a single factor.
        Returns: pd.Series with MultiIndex (datetime, instrument).
        """
        try:
            import io
            import sys as _sys
            from joblib import parallel_backend
            
            from .expr_parser import parse_expression, parse_symbol
            from . import function_lib as func_lib
            
            df = self.data_df.copy()
            
            expr = parse_symbol(factor_expression, df.columns)
            
            old_stdout = _sys.stdout
            _sys.stdout = io.StringIO()
            try:
                expr = parse_expression(expr)
            finally:
                _sys.stdout = old_stdout
            
            for col in sorted(df.columns, key=len, reverse=True):
                if col.startswith('$'):
                    expr = expr.replace(col, f"df['{col}']")
            
            exec_globals = {
                'df': df,
                'np': np,
                'pd': pd,
            }
            
            for name in dir(func_lib):
                if not name.startswith('_'):
                    obj = getattr(func_lib, name)
                    if callable(obj):
                        exec_globals[name] = obj
            
            with parallel_backend('threading', n_jobs=1):
                result = eval(expr, exec_globals)
            
            if isinstance(result, pd.DataFrame):
                result = result.iloc[:, 0]
            
            if isinstance(result, pd.Series):
                result.name = factor_name
                # Align result index with raw data (duplicate-safe)
                if not result.index.equals(df.index):
                    try:
                        if result.index.duplicated().any():
                            result = result[~result.index.duplicated(keep='last')]
                        result = result.reindex(df.index)
                    except Exception:
                        logger.debug(f"reindex fallback for [{factor_name}]")
                        result = result[~result.index.duplicated(keep='last')]
                        clean_idx = df.index[~df.index.duplicated(keep='last')]
                        result = result.reindex(clean_idx)
                return result.astype(np.float64)
            else:
                return pd.Series(result, index=df.index, name=factor_name).astype(np.float64)
                
        except Exception as e:
            logger.warning(f"Factor computation failed [{factor_name}]: {str(e)[:200]}")
            return None
    
    def calculate_factors_from_json(self, json_path: str, 
                                   max_factors: Optional[int] = None) -> pd.DataFrame:
        """Batch compute factors from JSON file."""
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        factors = data.get('factors', {})
        
        results = {}
        success_count = 0
        fail_count = 0
        
        factor_items = list(factors.items())
        if max_factors:
            factor_items = factor_items[:max_factors]
        
        total = len(factor_items)
        logger.debug(f"Computing {total} factors...")
        
        for i, (factor_id, factor_info) in enumerate(factor_items):
            factor_name = factor_info.get('factor_name', factor_id)
            factor_expr = factor_info.get('factor_expression', '')
            
            if not factor_expr:
                fail_count += 1
                continue
            
            if (i + 1) % 10 == 0 or i == 0:
                logger.debug(f"  Progress: {i+1}/{total}")
            
            result = self.calculate_factor(factor_name, factor_expr)
            
            if result is not None:
                results[factor_name] = result
                success_count += 1
            else:
                fail_count += 1
        
        print(f"Factor computation done: success {success_count}, failed {fail_count}")
        
        if results:
            return pd.DataFrame(results)
        return pd.DataFrame()
    
    def calculate_factors_batch(self, factors: List[Dict], use_cache: bool = True,
                                skip_compute: bool = False, persist_cache: bool = True,
                                return_diagnostics: bool = False):
        """
        Batch compute factors. Priority: 1) cache_location (result.h5),
        2) MD5 cache (factor_cache dir), 3) recompute from factor_expression
        (skipped when skip_compute=True). skip_compute=True skips cache misses.
        """
        import time as _time

        if use_cache and self.auto_extract_cache:
            self._auto_extract_cache_from_logs()

        diagnostics = FactorBatchDiagnostics(total_factors=len(factors))
        results: Dict[str, pd.Series] = {}
        failed_names: List[str] = []
        need_compute_factors = []

        for i, factor_info in enumerate(factors):
            factor_name = factor_info.get('factor_name', 'unknown')
            factor_expr = factor_info.get('factor_expression', '')
            cache_location = factor_info.get('cache_location')

            if not factor_expr:
                diagnostics.fail_count += 1
                diagnostics.factor_sources[factor_name] = 'failed'
                diagnostics.failed_factors.append(factor_name)
                failed_names.append(factor_name)
                continue

            result = None
            if use_cache and cache_location:
                h5_path = cache_location.get('result_h5_path', '')
                if h5_path:
                    if not Path(h5_path).exists():
                        diagnostics.missing_cache_locations.append({
                            'factor_name': factor_name,
                            'result_h5_path': h5_path,
                        })
                    result = self._load_from_cache_location(cache_location)
                    if result is not None:
                        diagnostics.cache_location_hit_count += 1
                        diagnostics.success_count += 1
                        diagnostics.factor_sources[factor_name] = 'h5_cache'
                        results[factor_name] = result
                        print(f"  [{i+1}/{len(factors)}] ✓ H5 cache: {factor_name}")
                        continue

            if use_cache:
                result = self._load_from_cache(factor_expr)
                if result is not None:
                    diagnostics.cache_hit_count += 1
                    diagnostics.success_count += 1
                    diagnostics.factor_sources[factor_name] = 'md5_cache'
                    results[factor_name] = result
                    print(f"  [{i+1}/{len(factors)}] ✓ MD5 cache: {factor_name}")
                    continue

            need_compute_factors.append((i, factor_info))
            print(f"  [{i+1}/{len(factors)}] ⏳ Pending: {factor_name}")

        if need_compute_factors:
            if skip_compute:
                diagnostics.skipped_count = len(need_compute_factors)
                skipped_names = [f.get('factor_name', 'unknown') for _, f in need_compute_factors]
                for skipped_name in skipped_names:
                    diagnostics.factor_sources[skipped_name] = 'skipped_uncached'
                print(f"  Skipping {diagnostics.skipped_count} uncached factors (skip_compute=True)")
                if skipped_names:
                    print(f"  Skipped: {', '.join(skipped_names)}")
            else:
                print(f"  Computing {len(need_compute_factors)} factors from expressions...")
                t0 = _time.time()
                computed_results, computed_failed_names, new_compute_count = self._calculate_factors_parallel(
                    need_compute_factors=need_compute_factors,
                    use_cache=use_cache,
                    persist_cache=persist_cache,
                )
                elapsed = _time.time() - t0
                results.update(computed_results)
                diagnostics.success_count += len(computed_results)
                diagnostics.fail_count += len(computed_failed_names)
                diagnostics.compute_count += new_compute_count
                failed_names.extend(computed_failed_names)
                diagnostics.failed_factors.extend(computed_failed_names)
                for name in computed_results:
                    diagnostics.factor_sources[name] = 'recomputed'
                for name in computed_failed_names:
                    diagnostics.factor_sources[name] = 'failed'
                print(
                    f"  Parallel recompute done: success {len(computed_results)}, "
                    f"failed {len(computed_failed_names)} ({elapsed:.1f}s)"
                )

        print(
            f"Factor load done: success {diagnostics.success_count}, failed {diagnostics.fail_count}, "
            f"skipped {diagnostics.skipped_count} | H5 cache {diagnostics.cache_location_hit_count}, "
            f"MD5 cache {diagnostics.cache_hit_count}, computed {diagnostics.compute_count}"
        )
        if failed_names:
            print(f"  Failed: {', '.join(failed_names)}")

        if not results:
            empty_df = pd.DataFrame()
            if return_diagnostics:
                return empty_df, diagnostics
            return empty_df

        aligned_results: Dict[str, pd.Series] = {}
        reference_index = None
        for name, series in results.items():
            if reference_index is None:
                reference_index = series.index
            validated = self._validate_and_align_result(series, name, reference_index)
            if validated is not None:
                aligned_results[name] = validated
            else:
                diagnostics.factor_sources[name] = 'dropped_invalid'

        diagnostics.aligned_result_count = len(aligned_results)
        result_df = pd.DataFrame(aligned_results) if aligned_results else pd.DataFrame()
        if not result_df.empty:
            logger.debug(f"  Result DataFrame: {result_df.shape}")

        if return_diagnostics:
            return result_df, diagnostics
        return result_df

    def _calculate_factors_parallel(
        self,
        need_compute_factors: List[Tuple[int, Dict]],
        use_cache: bool,
        persist_cache: bool,
    ) -> tuple[dict[str, pd.Series], list[str], int]:
        """Compute uncached factors in parallel."""
        factor_config = self._config.get('factor_calculation', {}) if self._config else {}
        backend = str(factor_config.get('recompute_backend', 'thread')).lower()
        n_jobs = int(factor_config.get('recompute_n_jobs', factor_config.get('n_jobs', 4)))

        tasks: list[tuple[str, str]] = []
        for _, factor_info in need_compute_factors:
            factor_name = factor_info.get('factor_name', 'unknown')
            factor_expr = factor_info.get('factor_expression', '')
            if factor_expr:
                tasks.append((factor_name, factor_expr))

        if not tasks:
            return {}, [], 0

        _ = self.data_df
        max_workers = max(1, min(n_jobs, len(tasks)))
        print(f"  Parallel recompute backend={backend}, workers={max_workers}")
        if backend == 'process' and max_workers > (os.cpu_count() or 1) * 2:
            logger.warning(
                "recompute_backend=process with workers=%s may be memory-heavy on large factor data",
                max_workers,
            )

        if backend == 'process':
            return self._calculate_factors_parallel_process(tasks, use_cache, persist_cache, max_workers)
        return self._calculate_factors_parallel_thread(tasks, use_cache, persist_cache, max_workers)

    def _calculate_factors_parallel_thread(
        self,
        tasks: List[tuple[str, str]],
        use_cache: bool,
        persist_cache: bool,
        max_workers: int,
    ) -> tuple[dict[str, pd.Series], list[str], int]:
        """Compute factors with threads to avoid duplicating the large source DataFrame."""
        results: dict[str, pd.Series] = {}
        failed_names: list[str] = []
        computed_count = 0

        if max_workers == 1:
            for factor_name, factor_expr in tasks:
                result = self.calculate_factor(factor_name, factor_expr)
                if result is not None and len(result) > 0 and not result.isna().all():
                    results[factor_name] = result
                    computed_count += 1
                    if use_cache and persist_cache:
                        self._save_to_cache(factor_expr, result)
                else:
                    failed_names.append(factor_name)
            return results, failed_names, computed_count

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_task = {
                executor.submit(self.calculate_factor, factor_name, factor_expr): (factor_name, factor_expr)
                for factor_name, factor_expr in tasks
            }
            for future in as_completed(future_to_task):
                factor_name, factor_expr = future_to_task[future]
                try:
                    result = future.result()
                except Exception as e:
                    print(f"  ✗ Error: {factor_name}: {str(e)[:80]}")
                    failed_names.append(factor_name)
                    continue

                if result is not None and len(result) > 0 and not result.isna().all():
                    results[factor_name] = result
                    computed_count += 1
                    print(f"  ✓ Computed: {factor_name}")
                    if use_cache and persist_cache:
                        self._save_to_cache(factor_expr, result)
                else:
                    print(f"  ✗ Failed: {factor_name}")
                    failed_names.append(factor_name)

        return results, failed_names, computed_count

    def _calculate_factors_parallel_process(
        self,
        tasks: List[tuple[str, str]],
        use_cache: bool,
        persist_cache: bool,
        max_workers: int,
    ) -> tuple[dict[str, pd.Series], list[str], int]:
        """Compute factors with processes."""
        results: dict[str, pd.Series] = {}
        failed_names: list[str] = []
        computed_count = 0

        data_df = self.data_df
        with tempfile.NamedTemporaryFile(prefix='qa_factor_data_', suffix='.pkl', delete=False) as tmp:
            data_path = tmp.name
        try:
            data_df.to_pickle(data_path)
            with ProcessPoolExecutor(
                max_workers=max_workers,
                initializer=_init_factor_compute_worker,
                initargs=(data_path,),
            ) as executor:
                future_to_task = {
                    executor.submit(_compute_factor_worker, task): task
                    for task in tasks
                }
                for future in as_completed(future_to_task):
                    factor_name, factor_expr = future_to_task[future]
                    try:
                        _, _, result, error = future.result()
                    except Exception as e:
                        print(f"  ✗ Error: {factor_name}: {str(e)[:80]}")
                        failed_names.append(factor_name)
                        continue

                    if error is not None:
                        print(f"  ✗ Error: {factor_name}: {error[:80]}")
                        failed_names.append(factor_name)
                        continue

                    if result is not None and len(result) > 0 and not result.isna().all():
                        results[factor_name] = result
                        computed_count += 1
                        print(f"  ✓ Computed: {factor_name}")
                        if use_cache and persist_cache:
                            self._save_to_cache(factor_expr, result)
                    else:
                        print(f"  ✗ Failed: {factor_name}")
                        failed_names.append(factor_name)
        finally:
            try:
                os.remove(data_path)
            except OSError:
                pass

        return results, failed_names, computed_count
    
    def _validate_and_align_result(self, result: pd.Series, factor_name: str, 
                                    reference_index: Optional[pd.Index] = None) -> Optional[pd.Series]:
        """Validate and align cached result index."""
        if result is None:
            return None
        
        target_idx = reference_index
        if target_idx is None:
            try:
                target_idx = self.data_df.index
            except Exception:
                return result if len(result) > 0 and not result.isna().all() else None
        
        # Align index (duplicate-safe)
        if not result.index.equals(target_idx):
            try:
                if result.index.duplicated().any():
                    result = result[~result.index.duplicated(keep='last')]
                if target_idx.duplicated().any():
                    target_idx = target_idx[~target_idx.duplicated(keep='last')]
                
                common_idx = result.index.intersection(target_idx)
                if len(common_idx) > len(target_idx) * 0.5:
                    result = result.reindex(target_idx)
                    logger.debug(f"    Index align: common {len(common_idx)}, target {len(target_idx)}")
                else:
                    logger.warning(f"    Cache index match rate too low ({len(common_idx)}/{len(target_idx)}), will recompute")
                    return None
            except Exception as e:
                logger.warning(f"    Index align failed: {e}, will recompute")
                return None
        
        # Validate data
        if result is None or len(result) == 0 or result.isna().all():
            return None
        
        return result


class CustomFactorDataLoader:
    """
    Converts computed factor values to Qlib-compatible format.
    """
    
    def __init__(self, factor_df: pd.DataFrame, label_expr: str = "Ref($close, -2) / Ref($close, -1) - 1"):
        self.factor_df = factor_df
        self.label_expr = label_expr
        
    def to_qlib_format(self, data_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Convert to Qlib data format."""
        from .expr_parser import parse_expression, parse_symbol
        from . import function_lib as func_lib
        
        df = data_df.copy()
        
        expr = parse_symbol(self.label_expr, df.columns)
        expr = parse_expression(expr)
        
        for col in sorted(df.columns, key=len, reverse=True):
            if col.startswith('$'):
                expr = expr.replace(col, f"df['{col}']")
        
        exec_globals = {'df': df, 'np': np, 'pd': pd}
        for name in dir(func_lib):
            if not name.startswith('_'):
                obj = getattr(func_lib, name)
                if callable(obj):
                    exec_globals[name] = obj
        
        label = eval(expr, exec_globals)
        if isinstance(label, pd.DataFrame):
            label = label.iloc[:, 0]
        
        labels_df = pd.DataFrame({'LABEL0': label})
        
        return self.factor_df, labels_df


def get_qlib_stock_data(config: Dict) -> pd.DataFrame:
    """Load stock data from Qlib."""
    import qlib
    from qlib.data import D
    
    data_config = config.get('data', {})
    
    # Prefer QLIB_DATA_DIR env (aligned with runner.py)
    provider_uri = (
        os.environ.get('QLIB_DATA_DIR')
        or os.environ.get('QLIB_PROVIDER_URI')
        or data_config.get('provider_uri', os.path.expanduser('~/.qlib/qlib_data/cn_data'))
    )
    provider_uri = os.path.expanduser(provider_uri)
    region = data_config.get('region', 'cn')
    
    try:
        qlib.init(provider_uri=provider_uri, region=region)
    except Exception:
        pass  # Already initialized
    
    start_time = data_config.get('start_time', '2016-01-01')
    end_time = data_config.get('end_time', '2025-12-31')
    market = data_config.get('market', 'csi300')
    
    stock_list = D.instruments(market)
    
    fields = ['$open', '$high', '$low', '$close', '$volume', '$vwap']
    df = D.features(
        stock_list,
        fields,
        start_time=start_time,
        end_time=end_time,
        freq='day'
    )
    
    df.columns = fields
    
    logger.debug(f"Loaded stock data: {len(df)} rows")
    
    return df


if __name__ == '__main__':
    """Test factor computation."""
    import yaml
    
    logging.basicConfig(level=logging.INFO)
    
    _project_root = Path(__file__).resolve().parents[2]
    config_path = _project_root / 'configs' / 'backtest.yaml'
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    print("Loading stock data...")
    data_df = get_qlib_stock_data(config)
    
    calculator = CustomFactorCalculator(data_df)
    
    test_expr = "RANK(-1 * TS_PCTCHANGE($close, 10))"
    print(f"\nTest expression: {test_expr}")
    
    result = calculator.calculate_factor("test_factor", test_expr)
    if result is not None:
        print(f"Success! Result shape: {result.shape}")
        print(result.head())
    else:
        print("Failed!")

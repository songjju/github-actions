"""
파일명: src/utils/logger.py
목적: 로또 예측 시스템의 통합 로깅 시스템
작성자: AI Assistant
작성일: 2025-08-30
버전: 1.0

주요 클래스/함수:
- SystemLogger: 통합 로깅 관리자
- PredictionLogger: 예측 전용 로거
- PerformanceLogger: 성능 측정 로거
- setup_logger: 로거 설정 함수
"""

import logging
import logging.handlers
import sys
import os
from pathlib import Path
from typing import Dict, Any, Optional, List
import json
from datetime import datetime, timedelta
import traceback
from contextlib import contextmanager
import time

# 상위 디렉토리 imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.utils.config import Config

class SystemLogger:
    """시스템 통합 로깅 관리자"""
    
    def __init__(self, name: str = "LottoPredictionSystem"):
        """
        초기화
        
        Args:
            name: 로거 이름
        """
        self.name = name
        self.logger = logging.getLogger(name)
        self.loggers = {}  # 서브 로거들
        self.log_dir = Config.LOGS_DIR
        self.setup_logging()
        
    def setup_logging(self):
        """로깅 시스템 설정"""
        # 로그 디렉토리 생성
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # 루트 로거 설정
        self.logger.setLevel(logging.DEBUG)
        
        # 기존 핸들러 제거
        for handler in self.logger.handlers[:]:
            self.logger.removeHandler(handler)
        
        # 포매터 설정
        detailed_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(module)s:%(funcName)s:%(lineno)d - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        
        simple_formatter = logging.Formatter(
            '%(asctime)s - %(levelname)s - %(message)s',
            datefmt='%H:%M:%S'
        )
        
        # 콘솔 핸들러
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(simple_formatter)
        self.logger.addHandler(console_handler)
        
        # 파일 핸들러 (전체 로그)
        file_handler = logging.handlers.RotatingFileHandler(
            self.log_dir / 'system.log',
            maxBytes=10*1024*1024,  # 10MB
            backupCount=5,
            encoding='utf-8'
        )
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(detailed_formatter)
        self.logger.addHandler(file_handler)
        
        # 에러 전용 핸들러
        error_handler = logging.handlers.RotatingFileHandler(
            self.log_dir / 'errors.log',
            maxBytes=5*1024*1024,  # 5MB
            backupCount=3,
            encoding='utf-8'
        )
        error_handler.setLevel(logging.ERROR)
        error_handler.setFormatter(detailed_formatter)
        self.logger.addHandler(error_handler)
        
        # 성능 로그 핸들러
        performance_handler = logging.FileHandler(
            self.log_dir / 'performance.log',
            encoding='utf-8'
        )
        performance_handler.setLevel(logging.INFO)
        performance_formatter = logging.Formatter(
            '%(asctime)s - PERFORMANCE - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        performance_handler.setFormatter(performance_formatter)
        
        # 성능 로거 별도 생성
        perf_logger = logging.getLogger(f"{self.name}.performance")
        perf_logger.addHandler(performance_handler)
        perf_logger.setLevel(logging.INFO)
        
        self.logger.info("로깅 시스템 초기화 완료")
    
    def get_logger(self, module_name: str) -> logging.Logger:
        """
        모듈별 로거 반환
        
        Args:
            module_name: 모듈 이름
            
        Returns:
            logging.Logger: 모듈 전용 로거
        """
        logger_name = f"{self.name}.{module_name}"
        if logger_name not in self.loggers:
            module_logger = logging.getLogger(logger_name)
            self.loggers[logger_name] = module_logger
            
        return self.loggers[logger_name]
    
    def log_system_info(self):
        """시스템 정보 로깅"""
        import platform
        import psutil
        
        self.logger.info("=== 시스템 정보 ===")
        self.logger.info(f"Python 버전: {platform.python_version()}")
        self.logger.info(f"플랫폼: {platform.platform()}")
        self.logger.info(f"CPU 개수: {psutil.cpu_count()}")
        self.logger.info(f"메모리: {psutil.virtual_memory().total // (1024**3)}GB")
        self.logger.info(f"로그 디렉토리: {self.log_dir}")
    
    def log_exception(self, exc: Exception, context: str = ""):
        """예외 상세 로깅"""
        self.logger.error(f"예외 발생 {context}: {type(exc).__name__}: {exc}")
        self.logger.error(f"스택 트레이스:\n{traceback.format_exc()}")
    
    def log_performance(self, operation: str, duration: float, details: Dict[str, Any] = None):
        """성능 정보 로깅"""
        perf_logger = logging.getLogger(f"{self.name}.performance")
        
        log_data = {
            'operation': operation,
            'duration_seconds': round(duration, 4),
            'timestamp': datetime.now().isoformat()
        }
        
        if details:
            log_data.update(details)
        
        perf_logger.info(json.dumps(log_data, ensure_ascii=False))
    
    @contextmanager
    def performance_timer(self, operation: str, **details):
        """성능 측정 컨텍스트 매니저"""
        start_time = time.time()
        try:
            yield
        finally:
            duration = time.time() - start_time
            self.log_performance(operation, duration, details)
    
    def get_log_stats(self) -> Dict[str, Any]:
        """로그 통계 반환"""
        stats = {}
        
        for log_file in ['system.log', 'errors.log', 'performance.log']:
            file_path = self.log_dir / log_file
            if file_path.exists():
                file_stats = file_path.stat()
                stats[log_file] = {
                    'size_mb': round(file_stats.st_size / (1024*1024), 2),
                    'modified': datetime.fromtimestamp(file_stats.st_mtime).isoformat(),
                    'exists': True
                }
            else:
                stats[log_file] = {'exists': False}
        
        return stats
    
    def cleanup_old_logs(self, days: int = 30):
        """오래된 로그 파일 정리"""
        cutoff_date = datetime.now() - timedelta(days=days)
        cleaned_count = 0
        
        for log_file in self.log_dir.glob('*.log*'):
            if log_file.stat().st_mtime < cutoff_date.timestamp():
                try:
                    log_file.unlink()
                    cleaned_count += 1
                    self.logger.info(f"오래된 로그 파일 삭제: {log_file.name}")
                except Exception as e:
                    self.logger.error(f"로그 파일 삭제 실패 {log_file.name}: {e}")
        
        self.logger.info(f"로그 정리 완료: {cleaned_count}개 파일 삭제")


class PredictionLogger:
    """예측 전용 로거"""
    
    def __init__(self, system_logger: SystemLogger):
        """
        초기화
        
        Args:
            system_logger: 시스템 로거
        """
        self.logger = system_logger.get_logger("prediction")
        self.prediction_log_file = Config.LOGS_DIR / 'predictions.jsonl'
        self.system_logger = system_logger
        
    def log_prediction_start(self, session_id: str, parameters: Dict[str, Any]):
        """예측 시작 로그"""
        self.logger.info(f"예측 세션 시작: {session_id}")
        self.logger.debug(f"예측 파라미터: {parameters}")
        
        # 상세 예측 로그를 JSON Lines 형식으로 저장
        log_entry = {
            'session_id': session_id,
            'event': 'prediction_start',
            'timestamp': datetime.now().isoformat(),
            'parameters': parameters
        }
        self._append_prediction_log(log_entry)
    
    def log_prediction_step(self, session_id: str, step: str, details: Dict[str, Any]):
        """예측 단계별 로그"""
        self.logger.debug(f"예측 단계 [{session_id}]: {step}")
        
        log_entry = {
            'session_id': session_id,
            'event': 'prediction_step',
            'step': step,
            'timestamp': datetime.now().isoformat(),
            'details': details
        }
        self._append_prediction_log(log_entry)
    
    def log_prediction_result(self, session_id: str, result: Dict[str, Any]):
        """예측 결과 로그"""
        prediction = result.get('final_prediction', [])
        confidence = result.get('ensemble_confidence', 0)
        
        self.logger.info(f"예측 완료 [{session_id}]: {prediction} (신뢰도: {confidence:.3f})")
        
        log_entry = {
            'session_id': session_id,
            'event': 'prediction_result',
            'timestamp': datetime.now().isoformat(),
            'result': result
        }
        self._append_prediction_log(log_entry)
    
    def log_prediction_error(self, session_id: str, error: Exception, context: str = ""):
        """예측 오류 로그"""
        self.logger.error(f"예측 오류 [{session_id}] {context}: {error}")
        self.system_logger.log_exception(error, f"예측 세션 {session_id}")
        
        log_entry = {
            'session_id': session_id,
            'event': 'prediction_error',
            'timestamp': datetime.now().isoformat(),
            'error': str(error),
            'context': context,
            'traceback': traceback.format_exc()
        }
        self._append_prediction_log(log_entry)
    
    def _append_prediction_log(self, log_entry: Dict[str, Any]):
        """예측 로그 파일에 추가"""
        try:
            with open(self.prediction_log_file, 'a', encoding='utf-8') as f:
                f.write(json.dumps(log_entry, ensure_ascii=False) + '\n')
        except Exception as e:
            self.logger.error(f"예측 로그 저장 실패: {e}")
    
    def get_prediction_history(self, limit: int = 100) -> List[Dict[str, Any]]:
        """예측 히스토리 조회"""
        history = []
        
        if not self.prediction_log_file.exists():
            return history
        
        try:
            with open(self.prediction_log_file, 'r', encoding='utf-8') as f:
                lines = f.readlines()
                
            # 최근 것부터 반환
            for line in reversed(lines[-limit:]):
                try:
                    log_entry = json.loads(line.strip())
                    history.append(log_entry)
                except json.JSONDecodeError:
                    continue
                    
        except Exception as e:
            self.logger.error(f"예측 히스토리 조회 실패: {e}")
        
        return history


class PerformanceLogger:
    """성능 측정 전용 로거"""
    
    def __init__(self, system_logger: SystemLogger):
        """
        초기화
        
        Args:
            system_logger: 시스템 로거
        """
        self.logger = system_logger.get_logger("performance")
        self.system_logger = system_logger
        self.performance_data = []
    
    def log_execution_time(self, function_name: str, execution_time: float, 
                          parameters: Dict[str, Any] = None):
        """실행 시간 로그"""
        self.logger.info(f"{function_name} 실행 시간: {execution_time:.4f}초")
        
        perf_entry = {
            'function': function_name,
            'execution_time': execution_time,
            'timestamp': datetime.now().isoformat(),
            'parameters': parameters or {}
        }
        
        self.performance_data.append(perf_entry)
        self.system_logger.log_performance(function_name, execution_time, parameters)
    
    def log_memory_usage(self, context: str, memory_mb: float):
        """메모리 사용량 로그"""
        self.logger.info(f"메모리 사용량 [{context}]: {memory_mb:.2f}MB")
        
        self.system_logger.log_performance(
            f"memory_usage_{context}", 
            memory_mb, 
            {'unit': 'MB', 'context': context}
        )
    
    def log_data_processing_stats(self, operation: str, 
                                 records_processed: int, 
                                 processing_time: float):
        """데이터 처리 통계 로그"""
        rate = records_processed / processing_time if processing_time > 0 else 0
        
        self.logger.info(
            f"데이터 처리 [{operation}]: {records_processed}건, "
            f"{processing_time:.2f}초, {rate:.1f}건/초"
        )
        
        self.system_logger.log_performance(
            f"data_processing_{operation}",
            processing_time,
            {
                'records_processed': records_processed,
                'processing_rate': rate
            }
        )
    
    @contextmanager
    def measure_performance(self, operation_name: str, **context):
        """성능 측정 컨텍스트 매니저"""
        import psutil
        import os
        
        # 시작 시점 측정
        start_time = time.time()
        process = psutil.Process(os.getpid())
        start_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        try:
            yield
        finally:
            # 종료 시점 측정
            end_time = time.time()
            end_memory = process.memory_info().rss / 1024 / 1024  # MB
            
            execution_time = end_time - start_time
            memory_delta = end_memory - start_memory
            
            # 로그 기록
            self.log_execution_time(operation_name, execution_time, context)
            
            if abs(memory_delta) > 1:  # 1MB 이상 변화시만 기록
                self.log_memory_usage(operation_name, memory_delta)
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """성능 요약 통계"""
        if not self.performance_data:
            return {"message": "성능 데이터가 없습니다."}
        
        # 함수별 통계
        function_stats = {}
        for entry in self.performance_data:
            func_name = entry['function']
            exec_time = entry['execution_time']
            
            if func_name not in function_stats:
                function_stats[func_name] = []
            function_stats[func_name].append(exec_time)
        
        # 요약 계산
        summary = {}
        for func_name, times in function_stats.items():
            summary[func_name] = {
                'count': len(times),
                'total_time': sum(times),
                'avg_time': sum(times) / len(times),
                'min_time': min(times),
                'max_time': max(times)
            }
        
        return {
            'summary': summary,
            'total_operations': len(self.performance_data),
            'report_time': datetime.now().isoformat()
        }


def setup_logger(name: str = "LottoPredictionSystem") -> SystemLogger:
    """
    로거 설정 편의 함수
    
    Args:
        name: 로거 이름
        
    Returns:
        SystemLogger: 설정된 시스템 로거
    """
    return SystemLogger(name)


def get_module_logger(module_name: str) -> logging.Logger:
    """
    모듈별 로거 반환 편의 함수
    
    Args:
        module_name: 모듈 이름
        
    Returns:
        logging.Logger: 모듈 전용 로거
    """
    # 글로벌 시스템 로거 사용
    if not hasattr(get_module_logger, '_system_logger'):
        get_module_logger._system_logger = setup_logger()
    
    return get_module_logger._system_logger.get_logger(module_name)


# === 데코레이터 ===
def log_execution_time(logger_name: str = None):
    """실행 시간 로깅 데코레이터"""
    def decorator(func):
        def wrapper(*args, **kwargs):
            logger = get_module_logger(logger_name or func.__module__)
            
            start_time = time.time()
            try:
                result = func(*args, **kwargs)
                execution_time = time.time() - start_time
                logger.info(f"{func.__name__} 실행 완료: {execution_time:.4f}초")
                return result
            except Exception as e:
                execution_time = time.time() - start_time
                logger.error(f"{func.__name__} 실행 실패: {execution_time:.4f}초, 오류: {e}")
                raise
        return wrapper
    return decorator


def log_function_call(logger_name: str = None):
    """함수 호출 로깅 데코레이터"""
    def decorator(func):
        def wrapper(*args, **kwargs):
            logger = get_module_logger(logger_name or func.__module__)
            
            # 인자 정보 (민감한 정보 제외)
            safe_args = []
            for arg in args:
                if isinstance(arg, (str, int, float, bool, list, dict)) and len(str(arg)) < 100:
                    safe_args.append(arg)
                else:
                    safe_args.append(f"<{type(arg).__name__}>")
            
            logger.debug(f"{func.__name__} 호출: args={safe_args}, kwargs={kwargs}")
            
            try:
                result = func(*args, **kwargs)
                logger.debug(f"{func.__name__} 완료")
                return result
            except Exception as e:
                logger.error(f"{func.__name__} 실패: {e}")
                raise
        return wrapper
    return decorator


# === 모듈 테스트 ===
if __name__ == "__main__":
    print("=== 로깅 시스템 테스트 ===")
    
    try:
        # 시스템 로거 초기화
        system_logger = setup_logger()
        system_logger.log_system_info()
        
        # 모듈별 로거 테스트
        data_logger = system_logger.get_logger("data_processing")
        prediction_logger_obj = PredictionLogger(system_logger)
        perf_logger = PerformanceLogger(system_logger)
        
        # 기본 로깅 테스트
        data_logger.info("데이터 로딩 테스트")
        data_logger.warning("테스트 경고 메시지")
        data_logger.debug("디버그 정보")
        
        # 예측 로깅 테스트
        session_id = "test_session_001"
        prediction_logger_obj.log_prediction_start(session_id, {"count": 6, "method": "ensemble"})
        prediction_logger_obj.log_prediction_step(session_id, "data_loading", {"records": 587})
        prediction_logger_obj.log_prediction_result(session_id, {
            "final_prediction": [1, 7, 14, 21, 28, 35],
            "ensemble_confidence": 0.75
        })
        
        # 성능 로깅 테스트
        with perf_logger.measure_performance("test_operation", test_param=123):
            time.sleep(0.1)  # 테스트용 지연
            
        perf_logger.log_data_processing_stats("csv_loading", 587, 0.05)
        
        # 예외 로깅 테스트
        try:
            raise ValueError("테스트 예외")
        except Exception as e:
            system_logger.log_exception(e, "테스트 컨텍스트")
        
        # 통계 출력
        print("\n로그 통계:")
        log_stats = system_logger.get_log_stats()
        for log_file, stats in log_stats.items():
            if stats.get('exists'):
                print(f"  {log_file}: {stats['size_mb']}MB")
        
        perf_summary = perf_logger.get_performance_summary()
        print(f"\n성능 요약: {perf_summary['total_operations']}개 작업 기록")
        
        print("✅ 로깅 시스템 테스트 완료")
        
    except Exception as e:
        print(f"❌ 테스트 중 오류: {e}")
        import traceback
        traceback.print_exc()
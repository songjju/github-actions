"""
파일명: src/utils/config.py
목적: 로또 예측 시스템의 전역 설정 및 상수 정의
작성자: AI Assistant
작성일: 2025-08-30
버전: 1.0

주요 클래스/함수:
- Config: 전역 설정 클래스
- LOTTO_CONSTANTS: 로또 관련 상수
"""

import os
from pathlib import Path
from typing import Dict, List, Any

class Config:
    """전역 설정 관리 클래스"""
    
    # === 프로젝트 경로 설정 ===
    PROJECT_ROOT = Path(__file__).parent.parent.parent
    DATA_DIR = PROJECT_ROOT / 'data'
    RAW_DATA_DIR = DATA_DIR / 'raw'  
    PROCESSED_DATA_DIR = DATA_DIR / 'processed'
    MODELS_DIR = DATA_DIR / 'models'
    OUTPUTS_DIR = PROJECT_ROOT / 'outputs'
    LOGS_DIR = PROJECT_ROOT / 'logs'
    
    # === 데이터 파일 경로 ===
    DATA_FILE_PATH = RAW_DATA_DIR / 'lotto_results.csv'
    PROCESSED_DATA_PATH = PROCESSED_DATA_DIR / 'cleaned_data.csv'
    STATISTICS_PATH = PROCESSED_DATA_DIR / 'statistical_features.csv'

    # === 로그 설정 ===
    LOG_LEVEL = 'INFO'
    LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    LOG_FILE = LOGS_DIR / 'lotto_prediction.log'
    
    # === 예측 시스템 설정 ===
    DEFAULT_PREDICTION_COUNT = 6
    MAX_PREDICTION_SETS = 100
    DEFAULT_CONFIDENCE_THRESHOLD = 0.6
    
    # === 다양성 보장 설정 ===
    DIVERSITY_SETTINGS = {
        'max_same_prediction_streak': 0,      # 동일 예측 연속 허용 횟수 (0=금지)
        'max_number_repeat_rate': 0.4,       # 번호별 최대 반복 비율
        'min_new_numbers_per_prediction': 2,  # 예측당 최소 신규 번호 수
        'recent_avoidance_window': 10,        # 최근 N회 예측 회피 창
        'force_diversity_after': 5,           # N회 후 강제 다양성 적용
        'overuse_threshold': 0.3,             # 과도 사용 임계치 비율
        'underuse_threshold': 0.3             # 과소 사용 임계치 비율
    }
    
    # === 천재적 공식 가중치 ===
    FORMULA_WEIGHTS = {
        'genius_insight': 0.20,          # 주 공식 (구현됨)
        'multi_dimensional': 0.15,       # 다차원 분석
        'creative_connection': 0.12,     # 창의적 연결
        'problem_redefinition': 0.10,    # 문제 재정의
        'innovative_solution': 0.08,     # 혁신적 솔루션
        'insight_amplification': 0.07,   # 통찰 증폭
        'thinking_evolution': 0.06,      # 사고 진화
        'complexity_solution': 0.05,     # 복잡성 솔루션
        'intuitive_leap': 0.07,          # 직관적 도약
        'integrated_wisdom': 0.10        # 통합 지혜
    }
    
    FORMULA_WEIGHTS_ALT = {
        'genius_insight': 0.25,          # 주 공식 비중 높임
        'integrated_wisdom': 0.20,       # 통합 지혜
        'multi_dimensional': 0.15,       
        'creative_connection': 0.12,     
        'problem_redefinition': 0.10,    
        'innovative_solution': 0.08,     
        'insight_amplification': 0.05,   
        'thinking_evolution': 0.03,      
        'complexity_solution': 0.02,     
        'intuitive_leap': 0.00           # 비활성화
    }

    # === 분석 파라미터 ===
    ANALYSIS_PARAMS = {
        'frequency_window': 52,           # 빈도 분석 기간 (주 단위)
        'pattern_detection_depth': 10,    # 패턴 탐지 깊이
        'statistical_confidence': 0.95,  # 통계적 신뢰도
        'trend_analysis_periods': [13, 26, 52],  # 트렌드 분석 기간들
        'correlation_threshold': 0.3      # 상관관계 임계치
    }
    
    # === 성능 최적화 설정 ===
    PERFORMANCE = {
        'cache_size': 1000,           # 캐시 크기
        'parallel_processing': True,   # 병렬 처리 활성화
        'max_workers': 4,             # 최대 워커 수
        'timeout_seconds': 30         # 계산 타임아웃
    }
    
    @classmethod
    def create_directories(cls):
        """필요한 디렉토리들을 생성"""
        directories = [
            cls.DATA_DIR,
            cls.RAW_DATA_DIR,
            cls.PROCESSED_DATA_DIR,
            cls.MODELS_DIR,
            cls.OUTPUTS_DIR,
            cls.OUTPUTS_DIR / 'predictions',
            cls.OUTPUTS_DIR / 'reports',
            cls.OUTPUTS_DIR / 'visualizations',
            cls.LOGS_DIR
        ]
        
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
    
    @classmethod
    def get_log_config(cls) -> Dict[str, Any]:
        """로깅 설정 반환"""
        return {
            'version': 1,
            'disable_existing_loggers': False,
            'formatters': {
                'standard': {
                    'format': cls.LOG_FORMAT
                },
            },
            'handlers': {
                'default': {
                    'level': cls.LOG_LEVEL,
                    'formatter': 'standard',
                    'class': 'logging.StreamHandler',
                },
                'file': {
                    'level': cls.LOG_LEVEL,
                    'formatter': 'standard',
                    'class': 'logging.FileHandler',
                    'filename': str(cls.LOG_FILE),
                    'mode': 'a',
                },
            },
            'loggers': {
                '': {
                    'handlers': ['default', 'file'],
                    'level': cls.LOG_LEVEL,
                    'propagate': False
                }
            }
        }


class LottoConstants:
    """로또 관련 상수"""
    
    # === 기본 로또 규칙 ===
    MIN_NUMBER = 1
    MAX_NUMBER = 45
    NUMBERS_TO_SELECT = 6
    BONUS_COUNT = 1
    TOTAL_BALLS = 45
    
    # === 당첨 등급 ===
    PRIZE_GRADES = {
        1: {'matched': 6, 'bonus': False, 'name': '1등'},
        2: {'matched': 5, 'bonus': True, 'name': '2등'},
        3: {'matched': 5, 'bonus': False, 'name': '3등'},
        4: {'matched': 4, 'bonus': False, 'name': '4등'},
        5: {'matched': 3, 'bonus': False, 'name': '5등'}
    }
    
    # === 확률 정보 ===
    THEORETICAL_ODDS = {
        1: 8145060,      # 1등 확률 (6개 모두 맞출 확률)
        2: 1357510,      # 2등 확률 (5개 + 보너스)
        3: 35724,        # 3등 확률 (5개)
        4: 733,          # 4등 확률 (4개)
        5: 45            # 5등 확률 (3개)
    }
    
    # === 번호 구간 정의 ===
    NUMBER_SECTIONS = {
        'low': (1, 15),      # 저구간
        'mid': (16, 30),     # 중구간  
        'high': (31, 45)     # 고구간
    }
    
    # === 패턴 분석 상수 ===
    PATTERN_TYPES = [
        'consecutive',       # 연속번호
        'odd_even',         # 홀짝
        'section_distribution',  # 구간분포
        'sum_range',        # 합계범위
        'gap_analysis'      # 간격분석
    ]


class AnalysisConstants:
    """분석 관련 상수"""
    
    # === 통계 분석 ===
    FREQUENCY_CATEGORIES = {
        'very_hot': 0.8,     # 매우 뜨거운 번호 (상위 20%)
        'hot': 0.6,          # 뜨거운 번호 (상위 40%)
        'normal': 0.4,       # 보통 번호 (중간 20%)
        'cold': 0.2,         # 차가운 번호 (하위 40%)
        'very_cold': 0.0     # 매우 차가운 번호 (하위 20%)
    }
    
    # === 패턴 임계치 ===
    PATTERN_THRESHOLDS = {
        'consecutive_max': 3,        # 최대 연속번호 개수
        'same_section_max': 4,       # 같은 구간 최대 개수
        'odd_even_balance': (2, 4),  # 홀짝 균형 범위
        'sum_normal_range': (100, 200)  # 정상 합계 범위
    }
    
    # === 시계열 분석 ===
    TIME_WINDOWS = {
        'short_term': 13,    # 단기 (분기)
        'medium_term': 26,   # 중기 (반년)
        'long_term': 52,     # 장기 (1년)
        'very_long_term': 104  # 초장기 (2년)
    }


class GeniusFormulaConstants:
    """천재적 공식 관련 상수"""
    
    # === 천재적 통찰 공식 (GI) 컴포넌트 범위 ===
    GI_COMPONENTS = {
        'observation_depth': (1, 10),      # 관찰의 깊이
        'connection_creativity': (1, 10),   # 연결의 독창성  
        'pattern_recognition': (1, 10),     # 패턴 인식 능력
        'synthetic_thinking': (1, 10),      # 종합적 사고
        'stereotype_level': (1, 10),        # 고정관념 수준
        'bias_degree': (1, 10)              # 편향 정도
    }
    
    # === 다차원 분석 가중치 ===
    DIMENSIONAL_WEIGHTS = {
        'temporal': 1.0,     # 시간적 차원
        'spatial': 0.8,      # 공간적 차원
        'abstract': 0.9,     # 추상적 차원
        'causal': 0.7,       # 인과적 차원
        'hierarchical': 0.6  # 계층적 차원
    }
    
    # === 창의적 연결 매개변수 ===
    CREATIVE_CONNECTION_PARAMS = {
        'intersection_weight': 1.0,   # 교집합 가중치
        'symmetric_diff_weight': 0.8, # 대칭차집합 가중치
        'function_mapping_weight': 1.2 # 함수 매핑 가중치
    }


# === 환경별 설정 ===
class DevelopmentConfig(Config):
    """개발 환경 설정"""
    DEBUG = True
    LOG_LEVEL = 'DEBUG'
    CACHE_ENABLED = False


class ProductionConfig(Config):
    """운영 환경 설정"""  
    DEBUG = False
    LOG_LEVEL = 'WARNING'
    CACHE_ENABLED = True
    PERFORMANCE = {
        **Config.PERFORMANCE,
        'max_workers': 8,
        'cache_size': 5000
    }


# === 설정 팩토리 ===
def get_config(env: str = 'development') -> Config:
    """환경에 따른 설정 반환
    
    Args:
        env (str): 환경 타입 ('development', 'production')
        
    Returns:
        Config: 설정 객체
    """
    configs = {
        'development': DevelopmentConfig,
        'production': ProductionConfig,
        'dev': DevelopmentConfig,
        'prod': ProductionConfig
    }
    
    return configs.get(env.lower(), DevelopmentConfig)


# === 검증 함수 ===
def validate_lotto_numbers(numbers: List[int]) -> bool:
    """로또 번호 유효성 검증
    
    Args:
        numbers (List[int]): 검증할 번호 목록
        
    Returns:
        bool: 유효성 여부
        
    Example:
        >>> validate_lotto_numbers([1, 5, 10, 15, 20, 25])
        True
        >>> validate_lotto_numbers([0, 1, 2, 3, 4, 5])  
        False
    """
    if not isinstance(numbers, list):
        return False
    
    if len(numbers) != LottoConstants.NUMBERS_TO_SELECT:
        return False
    
    if len(set(numbers)) != len(numbers):  # 중복 체크
        return False
    
    for num in numbers:
        if not isinstance(num, int):
            return False
        if not (LottoConstants.MIN_NUMBER <= num <= LottoConstants.MAX_NUMBER):
            return False
    
    return True


def get_number_section(number: int) -> str:
    """번호의 구간 반환
    
    Args:
        number (int): 로또 번호
        
    Returns:
        str: 구간 이름 ('low', 'mid', 'high')
    """
    if LottoConstants.NUMBER_SECTIONS['low'][0] <= number <= LottoConstants.NUMBER_SECTIONS['low'][1]:
        return 'low'
    elif LottoConstants.NUMBER_SECTIONS['mid'][0] <= number <= LottoConstants.NUMBER_SECTIONS['mid'][1]:
        return 'mid'
    else:
        return 'high'


# === 모듈 초기화 ===
if __name__ == "__main__":
    # 설정 테스트
    config = get_config('development')
    print(f"프로젝트 루트: {config.PROJECT_ROOT}")
    print(f"데이터 파일: {config.DATA_FILE_PATH}")
    print(f"로그 레벨: {config.LOG_LEVEL}")
    
    # 로또 번호 검증 테스트
    test_numbers = [1, 5, 10, 15, 20, 25]
    print(f"테스트 번호 {test_numbers} 유효성: {validate_lotto_numbers(test_numbers)}")
    
    # 구간 테스트
    for num in [5, 20, 35]:
        print(f"번호 {num}의 구간: {get_number_section(num)}")
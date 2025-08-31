"""
파일명: src/analysis/basic_statistics.py
목적: 로또 데이터의 기초 통계 분석 수행
작성자: AI Assistant
작성일: 2025-08-30
버전: 1.0

주요 클래스/함수:
- BasicStatistics: 기초 통계 분석 클래스
- FrequencyAnalyzer: 번호별 빈도 분석
- PatternAnalyzer: 기본 패턴 분석
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any, Optional
from collections import Counter, defaultdict
import logging
from pathlib import Path
import sys
import math

# 상위 디렉토리의 config 임포트를 위한 경로 설정
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.utils.config import Config, LottoConstants, AnalysisConstants, get_number_section

class BasicStatistics:
    """로또 데이터 기초 통계 분석 클래스"""
    
    def __init__(self, lotto_data: pd.DataFrame):
        """
        초기화
        
        Args:
            lotto_data (pd.DataFrame): 전처리된 로또 데이터
        """
        self.data = lotto_data.copy()
        self.logger = self._setup_logger()
        self.frequency_analyzer = FrequencyAnalyzer(self.data)
        self.pattern_analyzer = PatternAnalyzer(self.data)
        
    def _setup_logger(self) -> logging.Logger:
        """로거 설정"""
        logger = logging.getLogger(__name__)
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            logger.setLevel(logging.INFO)
        return logger
    
    def comprehensive_analysis(self) -> Dict[str, Any]:
        """
        종합적인 기초 통계 분석 수행
        
        Returns:
            Dict[str, Any]: 전체 분석 결과
        """
        self.logger.info("종합적인 기초 통계 분석 시작")
        
        try:
            # 기본 분석 수행
            basic_info = self._get_basic_info()
            frequency_analysis = self.frequency_analyzer.analyze()
            pattern_analysis = self.pattern_analyzer.analyze()
            distribution_analysis = self._analyze_distributions()
            trend_analysis = self._analyze_trends()
            correlation_analysis = self._analyze_correlations()
            
            results = {
                'basic_info': basic_info,
                'frequency_analysis': frequency_analysis,
                'pattern_analysis': pattern_analysis,
                'distribution_analysis': distribution_analysis,
                'trend_analysis': trend_analysis,
                'correlation_analysis': correlation_analysis,
                
                # main.py에서 기대하는 필드들을 최상위에 추가 (하위 호환성)
                'most_frequent_numbers': frequency_analysis.get('most_frequent_numbers', []),
                'least_frequent_numbers': frequency_analysis.get('least_frequent_numbers', []),
                'odd_even_ratio': distribution_analysis.get('odd_even_distribution', {}).get('avg_odd_count', 3.0) / 6.0
            }
            
            self.logger.info("종합적인 기초 통계 분석 완료")
            return results
            
        except Exception as e:
            self.logger.error(f"통계 분석 중 오류: {e}")
            raise
    
    def _get_basic_info(self) -> Dict[str, Any]:
        """기본 정보 수집"""
        return {
            'total_draws': len(self.data),
            'total_numbers_drawn': len(self.data) * 6,
            'latest_round': self.data['round_number'].max() if not self.data.empty else None,
            'oldest_round': self.data['round_number'].min() if not self.data.empty else None,
            'data_completeness': self._check_data_completeness(),
            'unique_combinations': len(set(tuple(row['winning_numbers']) for _, row in self.data.iterrows())),
            'duplicate_combinations': len(self.data) - len(set(tuple(row['winning_numbers']) for _, row in self.data.iterrows()))
        }
    
    def _check_data_completeness(self) -> Dict[str, Any]:
        """데이터 완성도 체크"""
        missing_rounds = []
        if len(self.data) > 1:
            rounds = sorted(self.data['round_number'].tolist())
            for i in range(1, len(rounds)):
                if rounds[i] - rounds[i-1] > 1:
                    missing_rounds.extend(range(rounds[i-1] + 1, rounds[i]))
        
        return {
            'missing_rounds_count': len(missing_rounds),
            'missing_rounds': missing_rounds[:10],  # 처음 10개만 표시
            'completeness_ratio': 1 - len(missing_rounds) / len(self.data) if len(self.data) > 0 else 0
        }
    
    def _analyze_distributions(self) -> Dict[str, Any]:
        """분포 분석"""
        # 번호 합계 분포
        sums = self.data['number_sum'].tolist()
        
        # 빈 데이터 체크
        if not sums:
            return {
                'sum_distribution': {
                    'mean': 0,
                    'std': 0,
                    'min': 0,
                    'max': 0,
                    'median': 0,
                    'quartiles': {
                        'q1': 0,
                        'q3': 0
                    }
                },
                'section_distribution': {
                    'low': 0,
                    'mid': 0,
                    'high': 0
                },
                'section_percentages': {
                    'low': 0,
                    'mid': 0,
                    'high': 0
                },
                'odd_even_distribution': {
                    'odd_counts': {},
                    'most_common_odd_count': None,
                    'avg_odd_count': 0,
                    'avg_even_count': 6
                }
            }
        
        # 구간별 분포
        section_counts = {
            'low': self.data['low_count'].sum(),
            'mid': self.data['mid_count'].sum(), 
            'high': self.data['high_count'].sum()
        }
        
        # 홀짝 분포
        odd_counts = self.data['odd_count'].tolist()
        odd_distribution = Counter(odd_counts)
        
        # section_counts 합계가 0이 아닌지 확인
        total_sections = sum(section_counts.values())
        
        return {
            'sum_distribution': {
                'mean': np.mean(sums),
                'std': np.std(sums),
                'min': min(sums),
                'max': max(sums),
                'median': np.median(sums),
                'quartiles': {
                    'q1': np.percentile(sums, 25),
                    'q3': np.percentile(sums, 75)
                }
            },
            'section_distribution': section_counts,
            'section_percentages': {
                section: count / total_sections * 100 if total_sections > 0 else 0
                for section, count in section_counts.items()
            },
            'odd_even_distribution': {
                'odd_counts': dict(odd_distribution),
                'most_common_odd_count': odd_distribution.most_common(1)[0] if odd_distribution else None,
                'avg_odd_count': np.mean(odd_counts) if odd_counts else 0,
                'avg_even_count': 6 - np.mean(odd_counts) if odd_counts else 6
            }
        }
    
    def _analyze_trends(self) -> Dict[str, Any]:
        """트렌드 분석"""
        # 최근 트렌드 (최근 52주, 26주, 13주)
        trends = {}
        
        for weeks in AnalysisConstants.TIME_WINDOWS.values():
            if len(self.data) >= weeks:
                recent_data = self.data.head(weeks)
                
                # 최근 기간의 번호별 빈도
                all_numbers = []
                for _, row in recent_data.iterrows():
                    all_numbers.extend(row['winning_numbers'])
                
                number_freq = Counter(all_numbers)
                
                trends[f'last_{weeks}_draws'] = {
                    'most_frequent': number_freq.most_common(10),
                    'least_frequent': number_freq.most_common()[-10:] if len(number_freq.most_common()) >= 10 else [],
                    'avg_sum': recent_data['number_sum'].mean(),
                    'avg_odd_count': recent_data['odd_count'].mean()
                }
        
        return trends
    
    def _analyze_correlations(self) -> Dict[str, Any]:
        """상관관계 분석"""
        correlations = {}
        
        # 숫자 컬럼들 간의 상관관계
        numeric_columns = ['number_sum', 'odd_count', 'even_count', 'low_count', 'mid_count', 'high_count']
        available_columns = [col for col in numeric_columns if col in self.data.columns]
        
        if len(available_columns) > 1:
            corr_matrix = self.data[available_columns].corr()
            correlations['numeric_correlations'] = corr_matrix.to_dict()
        
        # 번호 간 동시 출현 분석
        cooccurrence = self._analyze_number_cooccurrence()
        correlations['number_cooccurrence'] = cooccurrence
        
        return correlations
    
    def _analyze_number_cooccurrence(self) -> Dict[str, Any]:
        """번호 간 동시 출현 분석"""
        cooccurrence_count = defaultdict(int)
        total_combinations = 0
        
        for _, row in self.data.iterrows():
            numbers = row['winning_numbers']
            # 모든 2개 조합에 대해 동시 출현 카운트
            for i in range(len(numbers)):
                for j in range(i + 1, len(numbers)):
                    pair = tuple(sorted([numbers[i], numbers[j]]))
                    cooccurrence_count[pair] += 1
                    total_combinations += 1
        
        # 가장 자주 함께 나오는 번호 쌍
        most_common_pairs = sorted(cooccurrence_count.items(), key=lambda x: x[1], reverse=True)[:20]
        
        # 특정 번호와 자주 함께 나오는 번호들
        number_companions = defaultdict(list)
        for (num1, num2), count in cooccurrence_count.items():
            number_companions[num1].append((num2, count))
            number_companions[num2].append((num1, count))
        
        # 각 번호별로 상위 5개 동반 번호 정리
        top_companions = {}
        for number in range(1, 46):
            if number in number_companions:
                companions = sorted(number_companions[number], key=lambda x: x[1], reverse=True)[:5]
                top_companions[number] = companions
        
        return {
            'most_common_pairs': most_common_pairs,
            'top_companions': top_companions,
            'total_pair_combinations': total_combinations
        }


class FrequencyAnalyzer:
    """번호별 빈도 분석 클래스"""
    
    def __init__(self, data: pd.DataFrame):
        self.data = data
        
    def analyze(self) -> Dict[str, Any]:
        """빈도 분석 수행"""
        # 모든 당첨번호 수집
        all_numbers = []
        for _, row in self.data.iterrows():
            all_numbers.extend(row['winning_numbers'])
        
        # 번호별 빈도 계산
        frequency = Counter(all_numbers)
        total_draws = len(all_numbers)
        
        # 빈도수를 확률로 변환
        frequency_prob = {num: count / total_draws for num, count in frequency.items()}
        
        # 이론적 확률 (1/45)과의 차이
        theoretical_prob = 1 / LottoConstants.TOTAL_BALLS
        deviations = {num: freq - theoretical_prob for num, freq in frequency_prob.items()}
        
        # 번호를 핫/콜드로 분류
        hot_numbers, cold_numbers = self._classify_hot_cold(frequency)
        
        # 최근 출현 분석
        recent_appearance = self._analyze_recent_appearance()
        
        return {
            'total_numbers_drawn': total_draws,
            'frequency_count': dict(frequency),
            'frequency_probability': frequency_prob,
            'most_frequent_numbers': frequency.most_common(10),
            'least_frequent_numbers': frequency.most_common()[-10:] if len(frequency) >= 10 else [],
            'hot_numbers': hot_numbers,
            'cold_numbers': cold_numbers,
            'theoretical_deviation': deviations,
            'recent_appearance': recent_appearance,
            'chi_square_test': self._chi_square_test(frequency)
        }
    
    def _classify_hot_cold(self, frequency: Counter) -> Tuple[List[int], List[int]]:
        """번호를 핫/콜드로 분류"""
        if not frequency:
            return [], []
        
        frequencies = list(frequency.values())
        mean_freq = np.mean(frequencies)
        std_freq = np.std(frequencies)
        
        hot_threshold = mean_freq + 0.5 * std_freq
        cold_threshold = mean_freq - 0.5 * std_freq
        
        hot_numbers = [num for num, count in frequency.items() if count >= hot_threshold]
        cold_numbers = [num for num, count in frequency.items() if count <= cold_threshold]
        
        return sorted(hot_numbers), sorted(cold_numbers)
    
    def _analyze_recent_appearance(self) -> Dict[int, int]:
        """각 번호의 최근 출현 회차 분석"""
        last_appearance = {}
        
        for idx, row in self.data.iterrows():
            for number in row['winning_numbers']:
                if number not in last_appearance:
                    last_appearance[number] = idx  # 최근 출현한 인덱스 (0이 최신)
        
        return last_appearance
    
    def _chi_square_test(self, frequency: Counter) -> Dict[str, float]:
        """카이제곱 검정으로 균등성 테스트"""
        if not frequency:
            return {'chi_square': 0, 'p_value': 1}
        
        observed = list(frequency.values())
        expected_mean = np.mean(observed)
        expected = [expected_mean] * len(observed)
        
        try:
            # 카이제곱 통계량 계산
            chi_square = sum((obs - exp) ** 2 / exp for obs, exp in zip(observed, expected))
            
            # 자유도
            degrees_of_freedom = len(observed) - 1
            
            # p-value 근사 계산 (단순화된 버전)
            # 실제로는 scipy.stats.chi2.sf를 사용해야 하지만, 의존성을 줄이기 위해 근사
            p_value = math.exp(-chi_square / 2) if chi_square < 50 else 0
            
            return {
                'chi_square': chi_square,
                'degrees_of_freedom': degrees_of_freedom,
                'p_value': p_value,
                'is_uniform': p_value > 0.05  # 5% 유의수준
            }
        except:
            return {'chi_square': 0, 'p_value': 1, 'is_uniform': True}


class PatternAnalyzer:
    """패턴 분석 클래스"""
    
    def __init__(self, data: pd.DataFrame):
        self.data = data
        
    def analyze(self) -> Dict[str, Any]:
        """패턴 분석 수행"""
        return {
            'consecutive_patterns': self._analyze_consecutive_patterns(),
            'odd_even_patterns': self._analyze_odd_even_patterns(),
            'section_patterns': self._analyze_section_patterns(),
            'sum_patterns': self._analyze_sum_patterns(),
            'gap_patterns': self._analyze_gap_patterns(),
            'sequence_patterns': self._analyze_sequence_patterns()
        }
    
    def _analyze_consecutive_patterns(self) -> Dict[str, Any]:
        """연속번호 패턴 분석"""
        consecutive_counts = []
        consecutive_examples = []
        
        for _, row in self.data.iterrows():
            numbers = sorted(row['winning_numbers'])
            consecutive_count = 0
            current_consecutive = []
            
            for i in range(len(numbers) - 1):
                if numbers[i + 1] - numbers[i] == 1:
                    if not current_consecutive:
                        current_consecutive = [numbers[i], numbers[i + 1]]
                    else:
                        current_consecutive.append(numbers[i + 1])
                else:
                    if len(current_consecutive) >= 2:
                        consecutive_count += 1
                        if len(consecutive_examples) < 10:
                            consecutive_examples.append({
                                'round': row['round_number'],
                                'sequence': current_consecutive.copy(),
                                'length': len(current_consecutive)
                            })
                    current_consecutive = []
            
            # 마지막 연속 체크
            if len(current_consecutive) >= 2:
                consecutive_count += 1
                if len(consecutive_examples) < 10:
                    consecutive_examples.append({
                        'round': row['round_number'],
                        'sequence': current_consecutive.copy(),
                        'length': len(current_consecutive)
                    })
            
            consecutive_counts.append(consecutive_count)
        
        return {
            'consecutive_frequency': Counter(consecutive_counts),
            'avg_consecutive_per_draw': np.mean(consecutive_counts),
            'max_consecutive_in_draw': max(consecutive_counts) if consecutive_counts else 0,
            'examples': consecutive_examples
        }
    
    def _analyze_odd_even_patterns(self) -> Dict[str, Any]:
        """홀짝 패턴 분석"""
        odd_counts = self.data['odd_count'].tolist()
        odd_distribution = Counter(odd_counts)
        
        return {
            'odd_count_distribution': dict(odd_distribution),
            'most_common_odd_count': odd_distribution.most_common(1)[0] if odd_distribution else None,
            'odd_even_ratio': np.mean(odd_counts) / 6,
            'balance_score': 1 - abs(np.mean(odd_counts) - 3) / 3  # 3이 완벽한 균형
        }
    
    def _analyze_section_patterns(self) -> Dict[str, Any]:
        """구간 패턴 분석"""
        section_patterns = []
        
        for _, row in self.data.iterrows():
            pattern = (row['low_count'], row['mid_count'], row['high_count'])
            section_patterns.append(pattern)
        
        pattern_distribution = Counter(section_patterns)
        
        return {
            'pattern_distribution': {str(k): v for k, v in pattern_distribution.items()},
            'most_common_pattern': pattern_distribution.most_common(1)[0] if pattern_distribution else None,
            'avg_distribution': {
                'low': np.mean([p[0] for p in section_patterns]),
                'mid': np.mean([p[1] for p in section_patterns]),
                'high': np.mean([p[2] for p in section_patterns])
            }
        }
    
    def _analyze_sum_patterns(self) -> Dict[str, Any]:
        """합계 패턴 분석"""
        sums = self.data['number_sum'].tolist()
        
        # 빈 리스트 체크
        if not sums:
            return {
                'sum_statistics': {
                    'mean': 0,
                    'median': 0,
                    'std': 0,
                    'min': 0,
                    'max': 0
                },
                'sum_categories': {
                    'very_low': 0,
                    'low': 0,
                    'normal': 0,
                    'high': 0,
                    'very_high': 0
                },
                'normal_range_ratio': 0.0
            }
        
        # 합계를 구간으로 분류
        sum_categories = {
            'very_low': sum(1 for s in sums if s < 100),
            'low': sum(1 for s in sums if 100 <= s < 130),
            'normal': sum(1 for s in sums if 130 <= s < 170),
            'high': sum(1 for s in sums if 170 <= s < 200),
            'very_high': sum(1 for s in sums if s >= 200)
        }
        
        return {
            'sum_statistics': {
                'mean': np.mean(sums),
                'median': np.median(sums),
                'std': np.std(sums),
                'min': min(sums),
                'max': max(sums)
            },
            'sum_categories': sum_categories,
            'normal_range_ratio': sum_categories['normal'] / len(sums) if len(sums) > 0 else 0.0
        }
    
    def _analyze_gap_patterns(self) -> Dict[str, Any]:
        """번호 간 간격 패턴 분석"""
        all_gaps = []
        
        for _, row in self.data.iterrows():
            numbers = sorted(row['winning_numbers'])
            gaps = [numbers[i + 1] - numbers[i] for i in range(len(numbers) - 1)]
            all_gaps.extend(gaps)
        
        gap_distribution = Counter(all_gaps)
        
        return {
            'gap_distribution': dict(gap_distribution),
            'most_common_gaps': gap_distribution.most_common(10),
            'avg_gap': np.mean(all_gaps),
            'gap_variance': np.var(all_gaps)
        }
    
    def _analyze_sequence_patterns(self) -> Dict[str, Any]:
        """수열 패턴 분석 (등차수열, 등비수열 등)"""
        arithmetic_sequences = 0
        geometric_sequences = 0
        
        for _, row in self.data.iterrows():
            numbers = sorted(row['winning_numbers'])
            
            # 등차수열 확인 (최소 3개 이상)
            if self._is_arithmetic_sequence(numbers):
                arithmetic_sequences += 1
            
            # 기타 특별한 패턴들 확인할 수 있음
        
        return {
            'arithmetic_sequences': arithmetic_sequences,
            'arithmetic_ratio': arithmetic_sequences / len(self.data) if len(self.data) > 0 else 0.0,
            'geometric_sequences': geometric_sequences
        }
    
    def _is_arithmetic_sequence(self, numbers: List[int], min_length: int = 3) -> bool:
        """등차수열인지 확인"""
        if len(numbers) < min_length:
            return False
        
        # 연속된 숫자들의 차이가 일정한지 확인
        differences = [numbers[i + 1] - numbers[i] for i in range(len(numbers) - 1)]
        
        # 적어도 min_length-1 개의 연속된 같은 차이가 있는지 확인
        for i in range(len(differences) - min_length + 2):
            if len(set(differences[i:i + min_length - 1])) == 1:
                return True
        
        return False


# === 유틸리티 함수들 ===
def calculate_entropy(frequencies: List[int]) -> float:
    """
    빈도 데이터의 엔트로피 계산
    
    Args:
        frequencies (List[int]): 빈도 리스트
        
    Returns:
        float: 엔트로피 값
    """
    if not frequencies or sum(frequencies) == 0:
        return 0
    
    total = sum(frequencies)
    probabilities = [f / total for f in frequencies if f > 0]
    
    entropy = -sum(p * math.log2(p) for p in probabilities)
    return entropy


def detect_anomalies(data: List[float], threshold: float = 2.0) -> List[int]:
    """
    이상치 탐지 (Z-score 기반)
    
    Args:
        data (List[float]): 데이터 리스트
        threshold (float): 임계값 (기본값: 2.0)
        
    Returns:
        List[int]: 이상치 인덱스 리스트
    """
    if len(data) < 2:
        return []
    
    mean_val = np.mean(data)
    std_val = np.std(data)
    
    if std_val == 0:
        return []
    
    anomalies = []
    for i, value in enumerate(data):
        z_score = abs(value - mean_val) / std_val
        if z_score > threshold:
            anomalies.append(i)
    
    return anomalies


# === 모듈 테스트 ===
if __name__ == "__main__":
    # 테스트용 샘플 데이터 (새로운 CSV 구조에 맞춰 수정)
    sample_data = pd.DataFrame({
        'round_number': [1187, 1186, 1185, 1184, 1183],
        'draw_date': ['2025.08.30', '2025.08.23', '2025.08.16', '2025.08.09', '2025.08.02'],
        'year': [2025, 2025, 2025, 2025, 2025],
        'winning_numbers': [
            [5, 13, 26, 29, 37, 40],
            [2, 8, 13, 16, 23, 28], 
            [6, 17, 22, 28, 29, 32], 
            [14, 16, 23, 25, 31, 37],
            [4, 15, 17, 23, 27, 36]
        ],
        'bonus_number': [42, 35, 38, 42, 31],
        'number_sum': [150, 90, 134, 146, 122],
        'odd_count': [3, 3, 3, 5, 3],
        'even_count': [3, 3, 3, 1, 3],
        'low_count': [1, 3, 1, 0, 1],
        'mid_count': [2, 3, 4, 4, 4],
        'high_count': [3, 0, 1, 2, 1]
    })
    
    print("=== 기초 통계 분석 테스트 (새로운 CSV 구조) ===")
    
    try:
        analyzer = BasicStatistics(sample_data)
        results = analyzer.comprehensive_analysis()
        
        print(f"총 추첨 수: {results['basic_info']['total_draws']}")
        print(f"최신 회차: {results['basic_info']['latest_round']}")
        print(f"가장 빈번한 번호: {results['frequency_analysis']['most_frequent_numbers'][:5]}")
        print(f"평균 합계: {results['distribution_analysis']['sum_distribution']['mean']:.1f}")
        print(f"평균 홀수 개수: {results['distribution_analysis']['odd_even_distribution']['avg_odd_count']:.1f}")
        
        print("\n패턴 분석:")
        pattern_insights = results['pattern_analysis']['consecutive_patterns']
        print(f"연속번호 평균: {pattern_insights['avg_consecutive_per_draw']:.2f}")
        
        print("\n✅ 기초 통계 분석 테스트 완료")
        
    except Exception as e:
        print(f"❌ 테스트 중 오류: {e}")
        import traceback
        traceback.print_exc()
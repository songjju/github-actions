"""
파일명: src/analysis/basic_statistics.py
목적: 로또 데이터의 기초 통계 분석 수행 (오류 수정 버전)
작성자: AI Assistant
작성일: 2025-08-30
버전: 1.1

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
        """기본 정보 수집 - 안전한 winning_numbers 처리"""
        try:
            # winning_numbers 컬럼 처리
            unique_combinations = 0
            if 'winning_numbers' in self.data.columns:
                valid_combinations = []
                for _, row in self.data.iterrows():
                    winning_nums = row['winning_numbers']
                    
                    # 데이터 타입에 따른 처리
                    if isinstance(winning_nums, (list, tuple, np.ndarray)):
                        valid_combinations.append(tuple(winning_nums))
                    elif isinstance(winning_nums, str):
                        try:
                            # 문자열인 경우 파싱 시도
                            nums = [int(x.strip()) for x in winning_nums.strip('[]').split(',')]
                            if len(nums) == 6:
                                valid_combinations.append(tuple(nums))
                        except:
                            continue
                    elif isinstance(winning_nums, int):
                        # 개별 번호들로부터 조합 생성
                        nums = [
                            row.get('number_1', 0), row.get('number_2', 0),
                            row.get('number_3', 0), row.get('number_4', 0),
                            row.get('number_5', 0), row.get('number_6', 0)
                        ]
                        if all(isinstance(n, int) and 1 <= n <= 45 for n in nums):
                            valid_combinations.append(tuple(sorted(nums)))
                    
                unique_combinations = len(set(valid_combinations))
            
            return {
                'total_draws': len(self.data),
                'total_numbers_drawn': len(self.data) * 6,
                'latest_round': self.data['round_number'].max() if 'round_number' in self.data.columns and not self.data.empty else None,
                'oldest_round': self.data['round_number'].min() if 'round_number' in self.data.columns and not self.data.empty else None,
                'data_completeness': self._check_data_completeness(),
                'unique_combinations': unique_combinations,
                'duplicate_combinations': len(self.data) - unique_combinations if unique_combinations > 0 else 0
            }
        except Exception as e:
            self.logger.warning(f"기본 정보 수집 중 오류: {e}")
            # 오류 발생 시 기본값 반환
            return {
                'total_draws': len(self.data),
                'total_numbers_drawn': len(self.data) * 6,
                'latest_round': None,
                'oldest_round': None,
                'data_completeness': {'missing_rounds_count': 0, 'completeness_ratio': 1.0},
                'unique_combinations': 0,
                'duplicate_combinations': 0
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
        """트렌드 분석 - 안전한 winning_numbers 처리"""
        # 최근 트렌드 (최근 52주, 26주, 13주)
        trends = {}
        
        # AnalysisConstants.TIME_WINDOWS가 없는 경우 기본값 사용
        time_windows = getattr(AnalysisConstants, 'TIME_WINDOWS', {
            'quarterly': 13,
            'semi_annual': 26,
            'annual': 52
        })
        
        for period_name, weeks in time_windows.items():
            if len(self.data) >= weeks:
                recent_data = self.data.head(weeks)
                
                # 최근 기간의 번호별 빈도 - 안전한 처리
                all_numbers = []
                for _, row in recent_data.iterrows():
                    winning_nums = row['winning_numbers']
                    numbers = self._extract_numbers_safely(row, winning_nums)
                    if numbers:
                        all_numbers.extend(numbers)
                
                if all_numbers:
                    number_freq = Counter(all_numbers)
                    
                    trends[f'last_{weeks}_draws'] = {
                        'most_frequent': number_freq.most_common(10),
                        'least_frequent': number_freq.most_common()[-10:] if len(number_freq.most_common()) >= 10 else [],
                        'avg_sum': recent_data['number_sum'].mean() if 'number_sum' in recent_data.columns else sum(all_numbers) / len(recent_data),
                        'avg_odd_count': recent_data['odd_count'].mean() if 'odd_count' in recent_data.columns else sum(1 for n in all_numbers if n % 2 == 1) / len(recent_data)
                    }
                else:
                    trends[f'last_{weeks}_draws'] = {
                        'most_frequent': [],
                        'least_frequent': [],
                        'avg_sum': 0,
                        'avg_odd_count': 3
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
        
        # 번호 간 동시 출현 분석 (수정된 메서드 호출)
        cooccurrence = self._analyze_number_cooccurrence()
        correlations['number_cooccurrence'] = cooccurrence
        
        return correlations
    
    def _analyze_number_cooccurrence(self) -> Dict[str, Any]:
        """번호 간 동시 출현 분석 - 강화된 안전 처리"""
        cooccurrence_count = defaultdict(int)
        total_combinations = 0
        
        for idx, row in self.data.iterrows():
            winning_nums = row['winning_numbers']
            numbers = self._extract_numbers_safely(row, winning_nums)
            
            # 디버깅을 위한 타입 체크
            if not isinstance(numbers, (list, tuple, np.ndarray)):
                print(f"경고: 행 {idx}에서 numbers가 예상과 다른 타입: {type(numbers)}, 값: {numbers}")
                continue
                
            # 리스트로 변환
            if isinstance(numbers, np.ndarray):
                numbers = numbers.tolist()
            elif isinstance(numbers, tuple):
                numbers = list(numbers)
                
            # 길이 체크
            if len(numbers) != 6:
                continue
                
            # 모든 요소가 정수인지 확인
            if not all(isinstance(n, int) for n in numbers):
                continue
                
            # 모든 2개 조합에 대해 동시 출현 카운트
            try:
                for i in range(len(numbers)):
                    for j in range(i + 1, len(numbers)):
                        pair = tuple(sorted([numbers[i], numbers[j]]))
                        cooccurrence_count[pair] += 1
                        total_combinations += 1
            except Exception as e:
                print(f"경고: 행 {idx}에서 조합 생성 중 오류: {e}, numbers: {numbers}")
                continue
        
        if not cooccurrence_count:
            return {
                'most_common_pairs': [],
                'top_companions': {},
                'total_pair_combinations': 0,
                'average_cooccurrence': 0,
                'unique_pairs': 0
            }
        
        # 가장 자주 함께 나오는 번호 쌍 (상위 20개)
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
            'total_pair_combinations': total_combinations,
            'average_cooccurrence': total_combinations / len(cooccurrence_count) if cooccurrence_count else 0,
            'unique_pairs': len(cooccurrence_count)
        }

    def _extract_numbers_safely(self, row, winning_nums):
        """안전하게 당첨번호 추출 - 항상 리스트 반환 보장"""
        try:
            if isinstance(winning_nums, (list, tuple, np.ndarray)):
                result = list(winning_nums)
                # 모든 요소가 숫자인지 확인
                if all(isinstance(n, (int, float)) for n in result):
                    return [int(n) for n in result]
                
            elif isinstance(winning_nums, str):
                try:
                    nums = [int(x.strip()) for x in winning_nums.strip('[]').split(',')]
                    return nums if len(nums) == 6 else []
                except:
                    return []
                    
            elif isinstance(winning_nums, (int, float)):
                # 개별 번호 컬럼에서 추출
                nums = []
                for i in range(1, 7):
                    col_name = f'number_{i}'
                    if col_name in row and pd.notna(row[col_name]):
                        try:
                            nums.append(int(row[col_name]))
                        except:
                            continue
                return nums if len(nums) == 6 else []
            
            # 기타 경우: 개별 번호 컬럼들 시도
            nums = []
            for i in range(1, 7):
                col_name = f'number_{i}'
                if col_name in row and pd.notna(row[col_name]):
                    try:
                        nums.append(int(row[col_name]))
                    except:
                        continue
            
            return nums if len(nums) == 6 else []
            
        except Exception as e:
            print(f"_extract_numbers_safely 오류: {e}, winning_nums: {winning_nums}, type: {type(winning_nums)}")
            return []


class FrequencyAnalyzer:
    """번호별 빈도 분석 클래스"""
    
    def __init__(self, data: pd.DataFrame):
        self.data = data
        
    def analyze(self) -> Dict[str, Any]:
        """빈도 분석 수행 - 안전한 winning_numbers 처리"""
        # 모든 당첨번호 수집
        all_numbers = []
        
        for _, row in self.data.iterrows():
            winning_nums = row['winning_numbers']
            
            # 데이터 타입에 따른 안전한 처리
            if isinstance(winning_nums, (list, tuple, np.ndarray)):
                # 리스트/튜플/배열인 경우
                all_numbers.extend(winning_nums)
            elif isinstance(winning_nums, str):
                # 문자열인 경우 파싱 시도
                try:
                    nums = [int(x.strip()) for x in winning_nums.strip('[]').split(',')]
                    all_numbers.extend(nums)
                except:
                    continue
            elif isinstance(winning_nums, int):
                # int인 경우 개별 번호 컬럼들에서 수집
                nums = []
                for i in range(1, 7):
                    col_name = f'number_{i}'
                    if col_name in row and isinstance(row[col_name], int):
                        nums.append(row[col_name])
                
                # 유효한 번호들만 추가
                if len(nums) == 6 and all(1 <= n <= 45 for n in nums):
                    all_numbers.extend(nums)
            else:
                # 기타 경우: 개별 번호 컬럼들 시도
                try:
                    nums = [
                        row.get('number_1'), row.get('number_2'), row.get('number_3'),
                        row.get('number_4'), row.get('number_5'), row.get('number_6')
                    ]
                    # None이 아니고 유효한 숫자인 경우만
                    valid_nums = [n for n in nums if isinstance(n, int) and 1 <= n <= 45]
                    if len(valid_nums) == 6:
                        all_numbers.extend(valid_nums)
                except:
                    continue
        
        if not all_numbers:
            # 빈 결과 반환
            return {
                'frequency_count': {},
                'most_frequent_numbers': [],
                'least_frequent_numbers': [],
                'frequency_statistics': {
                    'total_numbers': 0,
                    'unique_numbers': 0,
                    'avg_frequency': 0,
                    'frequency_std': 0
                }
            }
        
        # 번호별 빈도 계산
        frequency = Counter(all_numbers)
        total_draws = len(all_numbers)
        
        # 빈도수를 확률로 변환
        frequency_prob = {num: count / total_draws for num, count in frequency.items()}
        
        # 가장 빈번한/드문 번호들
        most_frequent = sorted(frequency.items(), key=lambda x: x[1], reverse=True)[:10]
        least_frequent = sorted(frequency.items(), key=lambda x: x[1])[:10]
        
        # 통계 계산
        frequencies = list(frequency.values())
        
        return {
            'frequency_count': dict(frequency),
            'frequency_probability': frequency_prob,
            'most_frequent_numbers': [num for num, _ in most_frequent],
            'least_frequent_numbers': [num for num, _ in least_frequent],
            'frequency_statistics': {
                'total_numbers': total_draws,
                'unique_numbers': len(frequency),
                'avg_frequency': np.mean(frequencies) if frequencies else 0,
                'frequency_std': np.std(frequencies) if frequencies else 0,
                'max_frequency': max(frequencies) if frequencies else 0,
                'min_frequency': min(frequencies) if frequencies else 0
            }
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
            'number_gap_patterns': self._analyze_number_gaps(),
            'sum_patterns': self._analyze_sum_patterns(),
            'gap_patterns': self._analyze_gap_patterns(),
            'sequence_patterns': self._analyze_sequence_patterns()
        }
    
    def _analyze_consecutive_patterns(self) -> Dict[str, Any]:
        """연속번호 패턴 분석 - 안전한 winning_numbers 처리"""
        consecutive_counts = []
        
        for _, row in self.data.iterrows():
            winning_nums = row['winning_numbers']
            
            # 안전한 번호 추출
            numbers = self._extract_numbers_safely(row, winning_nums)
            if not numbers:
                continue
                
            numbers = sorted(numbers)
            
            # 연속 번호 카운트
            consecutive = 0
            current_streak = 1
            
            for i in range(1, len(numbers)):
                if numbers[i] - numbers[i-1] == 1:
                    current_streak += 1
                else:
                    if current_streak >= 2:
                        consecutive += current_streak
                    current_streak = 1
            
            if current_streak >= 2:
                consecutive += current_streak
                
            consecutive_counts.append(consecutive)
        
        if not consecutive_counts:
            return {
                'avg_consecutive_per_draw': 0,
                'max_consecutive': 0,
                'consecutive_frequency': {},
                'draws_with_consecutive': 0
            }
        
        return {
            'avg_consecutive_per_draw': np.mean(consecutive_counts),
            'max_consecutive': max(consecutive_counts),
            'consecutive_frequency': dict(Counter(consecutive_counts)),
            'draws_with_consecutive': sum(1 for c in consecutive_counts if c > 0)
        }
    
    def _analyze_odd_even_patterns(self) -> Dict[str, Any]:
        """홀짝 패턴 분석"""
        odd_counts = []
        
        for _, row in self.data.iterrows():
            winning_nums = row['winning_numbers']
            numbers = self._extract_numbers_safely(row, winning_nums)
            if not numbers:
                continue
                
            odd_count = sum(1 for n in numbers if n % 2 == 1)
            odd_counts.append(odd_count)
        
        if not odd_counts:
            return {
                'avg_odd_count': 3.0,
                'odd_count_distribution': {},
                'most_common_odd_count': [3, 0]
            }
        
        odd_distribution = Counter(odd_counts)
        most_common = odd_distribution.most_common(1)
        
        return {
            'avg_odd_count': np.mean(odd_counts),
            'odd_count_distribution': dict(odd_distribution),
            'most_common_odd_count': most_common[0] if most_common else [3, 0]
        }
    
    def _analyze_section_patterns(self) -> Dict[str, Any]:
        """구간 패턴 분석 (1-15: 저, 16-30: 중, 31-45: 고)"""
        section_counts = {'low': [], 'mid': [], 'high': []}
        
        for _, row in self.data.iterrows():
            winning_nums = row['winning_numbers']
            numbers = self._extract_numbers_safely(row, winning_nums)
            if not numbers:
                continue
            
            low_count = sum(1 for n in numbers if n <= 15)
            mid_count = sum(1 for n in numbers if 16 <= n <= 30) 
            high_count = sum(1 for n in numbers if n >= 31)
            
            section_counts['low'].append(low_count)
            section_counts['mid'].append(mid_count)
            section_counts['high'].append(high_count)
        
        if not section_counts['low']:
            return {
                'avg_distribution': {'low': 2.0, 'mid': 2.0, 'high': 2.0},
                'section_balance_score': 0.5
            }
        
        return {
            'avg_distribution': {
                'low': np.mean(section_counts['low']),
                'mid': np.mean(section_counts['mid']),
                'high': np.mean(section_counts['high'])
            },
            'section_balance_score': self._calculate_balance_score(section_counts)
        }
    
    def _analyze_number_gaps(self) -> Dict[str, Any]:
        """번호 간격 분석"""
        all_gaps = []
        
        for _, row in self.data.iterrows():
            winning_nums = row['winning_numbers']
            numbers = self._extract_numbers_safely(row, winning_nums)
            if not numbers:
                continue
                
            numbers = sorted(numbers)
            gaps = [numbers[i+1] - numbers[i] for i in range(len(numbers)-1)]
            all_gaps.extend(gaps)
        
        if not all_gaps:
            return {
                'avg_gap': 7.5,
                'gap_distribution': {},
                'most_common_gap': 1
            }
        
        gap_distribution = Counter(all_gaps)
        most_common = gap_distribution.most_common(1)
        
        return {
            'avg_gap': np.mean(all_gaps),
            'gap_distribution': dict(gap_distribution),
            'most_common_gap': most_common[0][0] if most_common else 1
        }
    
    def _analyze_sum_patterns(self) -> Dict[str, Any]:
        """합계 패턴 분석"""
        sums = []
        
        # number_sum 컬럼이 있으면 사용, 없으면 직접 계산
        if 'number_sum' in self.data.columns:
            sums = [s for s in self.data['number_sum'].tolist() if pd.notna(s)]
        else:
            # 직접 계산
            for _, row in self.data.iterrows():
                winning_nums = row['winning_numbers']
                numbers = self._extract_numbers_safely(row, winning_nums)
                if numbers:
                    sums.append(sum(numbers))
        
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
            winning_nums = row['winning_numbers']
            numbers = self._extract_numbers_safely(row, winning_nums)
            if not numbers:
                continue
                
            numbers = sorted(numbers)
            gaps = [numbers[i + 1] - numbers[i] for i in range(len(numbers) - 1)]
            all_gaps.extend(gaps)
        
        if not all_gaps:
            return {
                'gap_distribution': {},
                'most_common_gaps': [],
                'avg_gap': 7.5,
                'gap_variance': 0
            }
        
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
            winning_nums = row['winning_numbers']
            numbers = self._extract_numbers_safely(row, winning_nums)
            if not numbers:
                continue
                
            numbers = sorted(numbers)
            
            # 등차수열 확인 (최소 3개 이상)
            if self._is_arithmetic_sequence(numbers):
                arithmetic_sequences += 1
            
            # 기타 특별한 패턴들 확인할 수 있음
        
        return {
            'arithmetic_sequences': arithmetic_sequences,
            'arithmetic_ratio': arithmetic_sequences / len(self.data) if len(self.data) > 0 else 0.0,
            'geometric_sequences': geometric_sequences
        }
    
    def _extract_numbers_safely(self, row, winning_nums):
        """안전하게 당첨번호 추출"""
        if isinstance(winning_nums, (list, tuple, np.ndarray)):
            return list(winning_nums)
        elif isinstance(winning_nums, str):
            try:
                nums = [int(x.strip()) for x in winning_nums.strip('[]').split(',')]
                return nums if len(nums) == 6 else []
            except:
                return []
        elif isinstance(winning_nums, int):
            # 개별 번호 컬럼에서 추출
            nums = []
            for i in range(1, 7):
                col_name = f'number_{i}'
                if col_name in row and isinstance(row[col_name], int):
                    nums.append(row[col_name])
            return nums if len(nums) == 6 else []
        else:
            # 기타 경우: 개별 번호 컬럼들 시도
            try:
                nums = [
                    row.get('number_1'), row.get('number_2'), row.get('number_3'),
                    row.get('number_4'), row.get('number_5'), row.get('number_6')
                ]
                # None이 아니고 유효한 숫자인 경우만
                valid_nums = [n for n in nums if isinstance(n, int) and 1 <= n <= 45]
                return valid_nums if len(valid_nums) == 6 else []
            except:
                return []
    
    def _calculate_balance_score(self, section_counts):
        """구간 균형 점수 계산"""
        try:
            low_avg = np.mean(section_counts['low'])
            mid_avg = np.mean(section_counts['mid'])
            high_avg = np.mean(section_counts['high'])
            
            # 이상적인 균형은 각 구간에 2개씩
            ideal = 2.0
            deviations = [abs(low_avg - ideal), abs(mid_avg - ideal), abs(high_avg - ideal)]
            max_deviation = max(deviations)
            
            # 0에서 1 사이의 점수 (1이 완벽한 균형)
            return max(0, 1 - max_deviation / 3)
        except:
            return 0.5
    
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
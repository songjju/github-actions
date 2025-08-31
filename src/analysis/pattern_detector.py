"""
파일명: src/analysis/pattern_detector.py
목적: 로또 데이터의 다양한 패턴 탐지 및 분석
작성자: AI Assistant
작성일: 2025-08-31
버전: 1.0

주요 클래스/함수:
- LottoPatternDetector: 패턴 탐지 메인 클래스
- SequentialPatternDetector: 순차 패턴 탐지
- CyclicPatternDetector: 주기 패턴 탐지
- GeometricPatternDetector: 기하학적 패턴 탐지
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any, Optional, Set
from collections import Counter, defaultdict, deque
import math
import statistics
import logging
from datetime import datetime, timedelta
from pathlib import Path
import sys
from itertools import combinations, permutations
from scipy import stats
from scipy.fft import fft, fftfreq
import warnings

# 상위 디렉토리 imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.utils.config import Config, LottoConstants

warnings.filterwarnings('ignore')

class PatternResult:
    """패턴 탐지 결과를 담는 클래스"""
    
    def __init__(self, pattern_type: str, confidence: float):
        self.pattern_type = pattern_type
        self.confidence = confidence
        self.details = {}
        self.examples = []
        self.frequency = 0
        self.trend = 'stable'
        self.prediction_value = 0.0
        
    def to_dict(self) -> Dict[str, Any]:
        return {
            'pattern_type': self.pattern_type,
            'confidence': self.confidence,
            'details': self.details,
            'examples': self.examples,
            'frequency': self.frequency,
            'trend': self.trend,
            'prediction_value': self.prediction_value
        }


class LottoPatternDetector:
    """로또 패턴 탐지 메인 클래스"""
    
    def __init__(self, min_confidence: float = 0.1):
        """
        초기화
        
        Args:
            min_confidence (float): 최소 신뢰도 임계값
        """
        self.min_confidence = min_confidence
        self.logger = self._setup_logger()
        self.detected_patterns = []
        
        # 서브 탐지기들 초기화
        self.sequential_detector = SequentialPatternDetector()
        self.cyclic_detector = CyclicPatternDetector()
        self.geometric_detector = GeometricPatternDetector()
        self.statistical_detector = StatisticalPatternDetector()
        
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
    
    def detect_all_patterns(self, df: pd.DataFrame, 
                           include_weak: bool = False) -> Dict[str, List[PatternResult]]:
        """
        모든 패턴 탐지 수행
        
        Args:
            df (pd.DataFrame): 로또 데이터
            include_weak (bool): 약한 패턴도 포함할지 여부
            
        Returns:
            Dict[str, List[PatternResult]]: 패턴 타입별 결과
        """
        self.logger.info("전체 패턴 탐지 시작")
        
        results = {
            'sequential': [],
            'cyclic': [],
            'geometric': [],
            'statistical': [],
            'composite': []
        }
        
        try:
            # 1. 순차 패턴 탐지
            self.logger.info("순차 패턴 탐지")
            results['sequential'] = self.sequential_detector.detect_patterns(df)
            
            # 2. 주기 패턴 탐지
            self.logger.info("주기 패턴 탐지")
            results['cyclic'] = self.cyclic_detector.detect_patterns(df)
            
            # 3. 기하학적 패턴 탐지
            self.logger.info("기하학적 패턴 탐지")
            results['geometric'] = self.geometric_detector.detect_patterns(df)
            
            # 4. 통계적 패턴 탐지
            self.logger.info("통계적 패턴 탐지")
            results['statistical'] = self.statistical_detector.detect_patterns(df)
            
            # 5. 복합 패턴 탐지
            self.logger.info("복합 패턴 탐지")
            results['composite'] = self._detect_composite_patterns(df, results)
            
            # 신뢰도 필터링
            if not include_weak:
                for pattern_type in results:
                    results[pattern_type] = [
                        p for p in results[pattern_type] 
                        if p.confidence >= self.min_confidence
                    ]
            
            # 결과 정렬 (신뢰도 기준)
            for pattern_type in results:
                results[pattern_type].sort(key=lambda x: x.confidence, reverse=True)
            
            total_patterns = sum(len(patterns) for patterns in results.values())
            self.logger.info(f"패턴 탐지 완료: {total_patterns}개 패턴 발견")
            
            return results
            
        except Exception as e:
            self.logger.error(f"패턴 탐지 중 오류: {e}")
            raise
    
    def _detect_composite_patterns(self, df: pd.DataFrame, 
                                 individual_results: Dict) -> List[PatternResult]:
        """복합 패턴 탐지 (여러 패턴의 조합)"""
        composite_patterns = []
        
        # 시간-빈도 복합 패턴
        time_freq_pattern = self._detect_time_frequency_pattern(df)
        if time_freq_pattern:
            composite_patterns.append(time_freq_pattern)
        
        # 위치-값 복합 패턴
        position_value_pattern = self._detect_position_value_pattern(df)
        if position_value_pattern:
            composite_patterns.append(position_value_pattern)
        
        # 간격-주기 복합 패턴
        gap_cycle_pattern = self._detect_gap_cycle_pattern(df)
        if gap_cycle_pattern:
            composite_patterns.append(gap_cycle_pattern)
        
        return composite_patterns
    
    def _detect_time_frequency_pattern(self, df: pd.DataFrame) -> Optional[PatternResult]:
        """시간-빈도 복합 패턴 탐지"""
        try:
            number_cols = [f'num{i}' for i in range(1, 7)]
            if not all(col in df.columns for col in number_cols):
                return None
            
            # 각 번호의 시간대별 출현 빈도 분석
            if 'draw_date' in df.columns:
                df['draw_date'] = pd.to_datetime(df['draw_date'])
                df['month'] = df['draw_date'].dt.month
                df['quarter'] = df['draw_date'].dt.quarter
                
                seasonal_patterns = {}
                for number in range(1, 46):
                    monthly_freq = []
                    for month in range(1, 13):
                        month_data = df[df['month'] == month]
                        count = 0
                        for col in number_cols:
                            count += (month_data[col] == number).sum()
                        monthly_freq.append(count)
                    
                    # 계절성 강도 계산
                    if sum(monthly_freq) > 0:
                        seasonal_strength = np.std(monthly_freq) / np.mean(monthly_freq)
                        if seasonal_strength > 0.3:  # 30% 이상 변동
                            seasonal_patterns[number] = {
                                'strength': seasonal_strength,
                                'pattern': monthly_freq
                            }
                
                if seasonal_patterns:
                    pattern = PatternResult('time_frequency_composite', 0.4)
                    pattern.details = {
                        'seasonal_numbers': len(seasonal_patterns),
                        'patterns': seasonal_patterns
                    }
                    pattern.examples = list(seasonal_patterns.keys())[:5]
                    return pattern
        
        except Exception:
            pass
        
        return None
    
    def _detect_position_value_pattern(self, df: pd.DataFrame) -> Optional[PatternResult]:
        """위치-값 복합 패턴 탐지"""
        try:
            number_cols = [f'num{i}' for i in range(1, 7)]
            if not all(col in df.columns for col in number_cols):
                return None
            
            # 각 위치별 선호 번호 범위 분석
            position_preferences = {}
            
            for i, col in enumerate(number_cols):
                numbers = df[col].dropna()
                if len(numbers) > 10:
                    # 각 위치에서 특정 범위 선호도 계산
                    ranges = {
                        'low': (1, 15),
                        'medium': (16, 30), 
                        'high': (31, 45)
                    }
                    
                    range_counts = {}
                    for range_name, (start, end) in ranges.items():
                        count = ((numbers >= start) & (numbers <= end)).sum()
                        range_counts[range_name] = count / len(numbers)
                    
                    # 특정 범위에 50% 이상 집중되면 패턴으로 인정
                    max_range = max(range_counts, key=range_counts.get)
                    max_ratio = range_counts[max_range]
                    
                    if max_ratio > 0.5:
                        position_preferences[f'position_{i+1}'] = {
                            'preferred_range': max_range,
                            'concentration': max_ratio
                        }
            
            if position_preferences:
                confidence = sum(p['concentration'] for p in position_preferences.values()) / 6
                pattern = PatternResult('position_value_composite', confidence)
                pattern.details = position_preferences
                return pattern
        
        except Exception:
            pass
        
        return None
    
    def _detect_gap_cycle_pattern(self, df: pd.DataFrame) -> Optional[PatternResult]:
        """간격-주기 복합 패턴 탐지"""
        try:
            number_cols = [f'num{i}' for i in range(1, 7)]
            if not all(col in df.columns for col in number_cols):
                return None
            
            # 번호 간 간격의 주기성 분석
            gap_cycles = {}
            
            for idx, row in df.iterrows():
                numbers = [row[col] for col in number_cols if pd.notna(row[col])]
                if len(numbers) == 6:
                    sorted_numbers = sorted(numbers)
                    gaps = [sorted_numbers[i+1] - sorted_numbers[i] for i in range(5)]
                    
                    # 간격 패턴의 특성 분석
                    gap_pattern = tuple(gaps)
                    if gap_pattern not in gap_cycles:
                        gap_cycles[gap_pattern] = 0
                    gap_cycles[gap_pattern] += 1
            
            # 반복되는 간격 패턴 찾기
            frequent_patterns = [(pattern, count) for pattern, count in gap_cycles.items() 
                               if count >= 3]  # 3회 이상 반복
            
            if frequent_patterns:
                total_occurrences = sum(count for _, count in frequent_patterns)
                confidence = total_occurrences / len(df)
                
                pattern = PatternResult('gap_cycle_composite', confidence)
                pattern.details = {
                    'frequent_gap_patterns': frequent_patterns[:5],
                    'total_pattern_occurrences': total_occurrences
                }
                return pattern
        
        except Exception:
            pass
        
        return None
    
    def get_pattern_summary(self, results: Dict[str, List[PatternResult]]) -> Dict[str, Any]:
        """패턴 탐지 결과 요약"""
        summary = {
            'total_patterns': sum(len(patterns) for patterns in results.values()),
            'pattern_breakdown': {},
            'highest_confidence': 0.0,
            'most_frequent_type': '',
            'strong_patterns': 0,  # 신뢰도 0.5 이상
            'recommendations': []
        }
        
        for pattern_type, patterns in results.items():
            summary['pattern_breakdown'][pattern_type] = {
                'count': len(patterns),
                'avg_confidence': np.mean([p.confidence for p in patterns]) if patterns else 0,
                'max_confidence': max([p.confidence for p in patterns]) if patterns else 0
            }
            
            # 최고 신뢰도 업데이트
            if patterns:
                type_max = max([p.confidence for p in patterns])
                if type_max > summary['highest_confidence']:
                    summary['highest_confidence'] = type_max
        
        # 가장 많은 패턴 타입
        pattern_counts = [(ptype, len(patterns)) for ptype, patterns in results.items()]
        if pattern_counts:
            summary['most_frequent_type'] = max(pattern_counts, key=lambda x: x[1])[0]
        
        # 강한 패턴 수 계산
        for patterns in results.values():
            summary['strong_patterns'] += sum(1 for p in patterns if p.confidence >= 0.5)
        
        # 권장사항 생성
        summary['recommendations'] = self._generate_pattern_recommendations(results)
        
        return summary
    
    def _generate_pattern_recommendations(self, results: Dict) -> List[str]:
        """패턴 분석 기반 권장사항"""
        recommendations = []
        
        total_patterns = sum(len(patterns) for patterns in results.values())
        
        if total_patterns == 0:
            recommendations.append("명확한 패턴이 발견되지 않았습니다. 더 많은 데이터가 필요할 수 있습니다.")
        elif total_patterns < 5:
            recommendations.append("패턴의 수가 적습니다. 다른 분석 방법과 함께 사용하세요.")
        else:
            # 강한 패턴이 있는 경우
            strong_patterns = sum(1 for patterns in results.values() 
                                for p in patterns if p.confidence >= 0.5)
            if strong_patterns > 0:
                recommendations.append(f"{strong_patterns}개의 강한 패턴이 발견되었습니다. 예측에 활용하세요.")
            
            # 패턴 타입별 권장사항
            for pattern_type, patterns in results.items():
                if patterns and len(patterns) >= 3:
                    avg_conf = np.mean([p.confidence for p in patterns])
                    if avg_conf >= 0.3:
                        recommendations.append(f"{pattern_type} 패턴이 유용합니다 (평균 신뢰도: {avg_conf:.2f}).")
        
        return recommendations


class SequentialPatternDetector:
    """순차 패턴 탐지기"""
    
    def detect_patterns(self, df: pd.DataFrame) -> List[PatternResult]:
        """순차 패턴들 탐지"""
        patterns = []
        
        # 연속 번호 패턴
        consecutive_pattern = self._detect_consecutive_pattern(df)
        if consecutive_pattern:
            patterns.append(consecutive_pattern)
        
        # 등차수열 패턴
        arithmetic_pattern = self._detect_arithmetic_pattern(df)
        if arithmetic_pattern:
            patterns.append(arithmetic_pattern)
        
        # 피보나치 패턴
        fibonacci_pattern = self._detect_fibonacci_pattern(df)
        if fibonacci_pattern:
            patterns.append(fibonacci_pattern)
        
        # 거듭제곱 패턴
        power_pattern = self._detect_power_pattern(df)
        if power_pattern:
            patterns.append(power_pattern)
        
        return patterns
    
    def _detect_consecutive_pattern(self, df: pd.DataFrame) -> Optional[PatternResult]:
        """연속 번호 패턴 탐지"""
        try:
            number_cols = [f'num{i}' for i in range(1, 7)]
            if not all(col in df.columns for col in number_cols):
                return None
            
            consecutive_counts = {2: 0, 3: 0, 4: 0, 5: 0, 6: 0}
            total_draws = 0
            examples = []
            
            for idx, row in df.iterrows():
                numbers = [row[col] for col in number_cols if pd.notna(row[col])]
                if len(numbers) == 6:
                    sorted_numbers = sorted(numbers)
                    total_draws += 1
                    
                    # 연속 번호 찾기
                    max_consecutive = self._find_max_consecutive(sorted_numbers)
                    if max_consecutive >= 2:
                        consecutive_counts[max_consecutive] += 1
                        if len(examples) < 5:
                            examples.append((idx, sorted_numbers, max_consecutive))
            
            if total_draws > 0:
                # 연속 번호가 있는 비율
                has_consecutive = sum(count for length, count in consecutive_counts.items() 
                                    if length >= 2)
                consecutive_ratio = has_consecutive / total_draws
                
                if consecutive_ratio > 0.1:  # 10% 이상
                    pattern = PatternResult('consecutive_numbers', consecutive_ratio)
                    pattern.details = {
                        'consecutive_distribution': consecutive_counts,
                        'total_with_consecutive': has_consecutive,
                        'consecutive_ratio': consecutive_ratio
                    }
                    pattern.examples = examples
                    pattern.frequency = has_consecutive
                    return pattern
        
        except Exception:
            pass
        
        return None
    
    def _find_max_consecutive(self, sorted_numbers: List[int]) -> int:
        """정렬된 번호에서 최대 연속 길이 찾기"""
        max_consecutive = 1
        current_consecutive = 1
        
        for i in range(1, len(sorted_numbers)):
            if sorted_numbers[i] == sorted_numbers[i-1] + 1:
                current_consecutive += 1
            else:
                max_consecutive = max(max_consecutive, current_consecutive)
                current_consecutive = 1
        
        max_consecutive = max(max_consecutive, current_consecutive)
        return max_consecutive
    
    def _detect_arithmetic_pattern(self, df: pd.DataFrame) -> Optional[PatternResult]:
        """등차수열 패턴 탐지"""
        try:
            number_cols = [f'num{i}' for i in range(1, 7)]
            if not all(col in df.columns for col in number_cols):
                return None
            
            arithmetic_sequences = []
            
            for idx, row in df.iterrows():
                numbers = [row[col] for col in number_cols if pd.notna(row[col])]
                if len(numbers) == 6:
                    sorted_numbers = sorted(numbers)
                    
                    # 3개 이상의 등차수열 찾기
                    for start_idx in range(len(sorted_numbers) - 2):
                        for end_idx in range(start_idx + 2, len(sorted_numbers)):
                            subseq = sorted_numbers[start_idx:end_idx + 1]
                            if self._is_arithmetic_sequence(subseq):
                                arithmetic_sequences.append((idx, subseq))
                                break
            
            if len(arithmetic_sequences) >= 3:  # 최소 3개 회차에서 발견
                confidence = len(arithmetic_sequences) / len(df)
                pattern = PatternResult('arithmetic_sequence', confidence)
                pattern.details = {
                    'sequences_found': len(arithmetic_sequences),
                    'common_differences': self._analyze_common_differences(arithmetic_sequences)
                }
                pattern.examples = arithmetic_sequences[:5]
                pattern.frequency = len(arithmetic_sequences)
                return pattern
        
        except Exception:
            pass
        
        return None
    
    def _is_arithmetic_sequence(self, numbers: List[int]) -> bool:
        """등차수열인지 확인"""
        if len(numbers) < 3:
            return False
        
        diff = numbers[1] - numbers[0]
        for i in range(2, len(numbers)):
            if numbers[i] - numbers[i-1] != diff:
                return False
        return True
    
    def _analyze_common_differences(self, sequences: List[Tuple]) -> Dict:
        """공차 분석"""
        differences = Counter()
        for _, seq in sequences:
            if len(seq) >= 3:
                diff = seq[1] - seq[0]
                differences[diff] += 1
        
        return {
            'most_common_differences': differences.most_common(3),
            'total_different_diffs': len(differences)
        }
    
    def _detect_fibonacci_pattern(self, df: pd.DataFrame) -> Optional[PatternResult]:
        """피보나치 패턴 탐지"""
        try:
            # 45 이하 피보나치 수들
            fib_numbers = [1, 1, 2, 3, 5, 8, 13, 21, 34]
            
            number_cols = [f'num{i}' for i in range(1, 7)]
            if not all(col in df.columns for col in number_cols):
                return None
            
            fibonacci_occurrences = 0
            total_numbers = 0
            examples = []
            
            for idx, row in df.iterrows():
                numbers = [row[col] for col in number_cols if pd.notna(row[col])]
                fib_count = sum(1 for num in numbers if num in fib_numbers)
                total_numbers += len(numbers)
                
                if fib_count >= 2:  # 2개 이상의 피보나치 수
                    fibonacci_occurrences += 1
                    if len(examples) < 5:
                        fib_nums = [num for num in numbers if num in fib_numbers]
                        examples.append((idx, fib_nums))
            
            if fibonacci_occurrences >= 3:
                confidence = fibonacci_occurrences / len(df)
                pattern = PatternResult('fibonacci_numbers', confidence)
                pattern.details = {
                    'draws_with_fibonacci': fibonacci_occurrences,
                    'fibonacci_density': fibonacci_occurrences / len(df)
                }
                pattern.examples = examples
                pattern.frequency = fibonacci_occurrences
                return pattern
        
        except Exception:
            pass
        
        return None
    
    def _detect_power_pattern(self, df: pd.DataFrame) -> Optional[PatternResult]:
        """거듭제곱 패턴 탐지"""
        try:
            # 45 이하 완전제곱수들
            perfect_squares = [1, 4, 9, 16, 25, 36]
            
            number_cols = [f'num{i}' for i in range(1, 7)]
            if not all(col in df.columns for col in number_cols):
                return None
            
            power_occurrences = 0
            examples = []
            
            for idx, row in df.iterrows():
                numbers = [row[col] for col in number_cols if pd.notna(row[col])]
                square_count = sum(1 for num in numbers if num in perfect_squares)
                
                if square_count >= 2:  # 2개 이상의 완전제곱수
                    power_occurrences += 1
                    if len(examples) < 5:
                        squares = [num for num in numbers if num in perfect_squares]
                        examples.append((idx, squares))
            
            if power_occurrences >= 3:
                confidence = power_occurrences / len(df)
                pattern = PatternResult('perfect_squares', confidence)
                pattern.details = {
                    'draws_with_squares': power_occurrences,
                    'square_density': power_occurrences / len(df)
                }
                pattern.examples = examples
                pattern.frequency = power_occurrences
                return pattern
        
        except Exception:
            pass
        
        return None


class CyclicPatternDetector:
    """주기 패턴 탐지기"""
    
    def detect_patterns(self, df: pd.DataFrame) -> List[PatternResult]:
        """주기 패턴들 탐지"""
        patterns = []
        
        # 시간 기반 주기 패턴
        time_cycle_pattern = self._detect_time_cycle_pattern(df)
        if time_cycle_pattern:
            patterns.append(time_cycle_pattern)
        
        # 번호 출현 주기 패턴
        number_cycle_pattern = self._detect_number_cycle_pattern(df)
        if number_cycle_pattern:
            patterns.append(number_cycle_pattern)
        
        # 간격 주기 패턴
        gap_cycle_pattern = self._detect_gap_cycle_pattern(df)
        if gap_cycle_pattern:
            patterns.append(gap_cycle_pattern)
        
        return patterns
    
    def _detect_time_cycle_pattern(self, df: pd.DataFrame) -> Optional[PatternResult]:
        """시간 기반 주기 패턴 탐지"""
        try:
            if 'draw_date' not in df.columns:
                return None
            
            df['draw_date'] = pd.to_datetime(df['draw_date'])
            df['day_of_week'] = df['draw_date'].dt.dayofweek
            df['week_of_month'] = df['draw_date'].dt.day // 7 + 1
            
            # 요일별 패턴 분석
            dow_patterns = {}
            number_cols = [f'num{i}' for i in range(1, 7)]
            
            for dow in range(7):
                dow_data = df[df['day_of_week'] == dow]
                if len(dow_data) > 5:  # 최소 5회 이상
                    # 해당 요일의 번호 분포 특성
                    all_numbers = []
                    for col in number_cols:
                        if col in dow_data.columns:
                            all_numbers.extend(dow_data[col].dropna().tolist())
                    
                    if all_numbers:
                        # 평균, 표준편차 계산
                        mean_num = np.mean(all_numbers)
                        std_num = np.std(all_numbers)
                        
                        dow_patterns[dow] = {
                            'count': len(dow_data),
                            'mean_number': mean_num,
                            'std_number': std_num,
                            'number_distribution': Counter(all_numbers).most_common(5)
                        }
            
            if len(dow_patterns) >= 2:
                # 요일별 차이의 유의성 검정
                means = [p['mean_number'] for p in dow_patterns.values()]
                if len(means) > 1 and np.std(means) / np.mean(means) > 0.1:
                    confidence = min(0.8, np.std(means) / np.mean(means))
                    
                    pattern = PatternResult('time_cycle', confidence)
                    pattern.details = {
                        'day_patterns': dow_patterns,
                        'pattern_strength': np.std(means) / np.mean(means)
                    }
                    return pattern
        
        except Exception:
            pass
        
        return None
    
    def _detect_number_cycle_pattern(self, df: pd.DataFrame) -> Optional[PatternResult]:
        """번호 출현 주기 패턴 탐지"""
        try:
            number_cols = [f'num{i}' for i in range(1, 7)]
            if not all(col in df.columns for col in number_cols):
                return None
            
            # 각 번호의 출현 간격 분석
            number_cycles = {}
            
            for number in range(1, 46):
                appearances = []
                for idx, row in df.iterrows():
                    if any(row[col] == number for col in number_cols if pd.notna(row[col])):
                        appearances.append(idx)
                
                if len(appearances) >= 3:
                    # 출현 간격 계산
                    gaps = [appearances[i+1] - appearances[i] for i in range(len(appearances)-1)]
                    
                    # 주기성 확인 (간격의 일관성)
                    if len(gaps) > 1:
                        gap_std = np.std(gaps)
                        gap_mean = np.mean(gaps)
                        
                        if gap_mean > 0:
                            consistency = 1 - (gap_std / gap_mean)
                            
                            if consistency > 0.5:  # 50% 이상 일관성
                                number_cycles[number] = {
                                    'average_gap': gap_mean,
                                    'consistency': consistency,
                                    'appearances': len(appearances),
                                    'gaps': gaps
                                }
            
            if len(number_cycles) >= 5:  # 5개 이상 번호에서 주기성 발견
                avg_consistency = np.mean([cycle['consistency'] for cycle in number_cycles.values()])
                
                pattern = PatternResult('number_cycle', avg_consistency)
                pattern.details = {
                    'cyclic_numbers': len(number_cycles),
                    'average_consistency': avg_consistency,
                    'cycles': number_cycles
                }
                pattern.frequency = len(number_cycles)
                return pattern
        
        except Exception:
            pass
        
        return None
    
    def _detect_gap_cycle_pattern(self, df: pd.DataFrame) -> Optional[PatternResult]:
        """간격 주기 패턴 탐지"""
        try:
            number_cols = [f'num{i}' for i in range(1, 7)]
            if not all(col in df.columns for col in number_cols):
                return None
            
            gap_sequences = []
            
            for idx, row in df.iterrows():
                numbers = [row[col] for col in number_cols if pd.notna(row[col])]
                if len(numbers) == 6:
                    sorted_numbers = sorted(numbers)
                    gaps = [sorted_numbers[i+1] - sorted_numbers[i] for i in range(5)]
                    gap_sequences.append(gaps)
            
            if len(gap_sequences) >= 10:
                # 간격 패턴의 주기성 분석
                gap_patterns = Counter()
                for gaps in gap_sequences:
                    pattern_key = tuple(gaps)
                    gap_patterns[pattern_key] += 1
                
                # 반복되는 패턴 찾기
                repeated_patterns = [(pattern, count) for pattern, count in gap_patterns.items() 
                                   if count >= 2]
                
                if repeated_patterns:
                    total_repeated = sum(count for _, count in repeated_patterns)
                    confidence = total_repeated / len(gap_sequences)
                    
                    pattern = PatternResult('gap_cycle', confidence)
                    pattern.details = {
                        'repeated_gap_patterns': repeated_patterns[:10],
                        'pattern_repetition_rate': confidence
                    }
                    pattern.frequency = total_repeated
                    return pattern
        
        except Exception:
            pass
        
        return None


class GeometricPatternDetector:
    """기하학적 패턴 탐지기"""
    
    def detect_patterns(self, df: pd.DataFrame) -> List[PatternResult]:
        """기하학적 패턴들 탐지"""
        patterns = []
        
        # 황금비 패턴
        golden_ratio_pattern = self._detect_golden_ratio_pattern(df)
        if golden_ratio_pattern:
            patterns.append(golden_ratio_pattern)
        
        # 대칭 패턴
        symmetry_pattern = self._detect_symmetry_pattern(df)
        if symmetry_pattern:
            patterns.append(symmetry_pattern)
        
        # 비율 패턴
        ratio_pattern = self._detect_ratio_pattern(df)
        if ratio_pattern:
            patterns.append(ratio_pattern)
        
        return patterns
    
    def _detect_golden_ratio_pattern(self, df: pd.DataFrame) -> Optional[PatternResult]:
        """황금비 패턴 탐지"""
        try:
            number_cols = [f'num{i}' for i in range(1, 7)]
            if not all(col in df.columns for col in number_cols):
                return None
            
            golden_ratio = (1 + math.sqrt(5)) / 2  # 약 1.618
            golden_pairs = []
            
            for idx, row in df.iterrows():
                numbers = [row[col] for col in number_cols if pd.notna(row[col])]
                if len(numbers) >= 2:
                    # 모든 번호 쌍에 대해 비율 확인
                    for i in range(len(numbers)):
                        for j in range(i+1, len(numbers)):
                            larger = max(numbers[i], numbers[j])
                            smaller = min(numbers[i], numbers[j])
                            
                            if smaller > 0:
                                ratio = larger / smaller
                                # 황금비와 10% 내외 차이
                                if abs(ratio - golden_ratio) / golden_ratio < 0.1:
                                    golden_pairs.append((idx, smaller, larger, ratio))
            
            if len(golden_pairs) >= 5:
                confidence = len(golden_pairs) / (len(df) * 15)  # 15는 가능한 쌍의 수
                
                pattern = PatternResult('golden_ratio', confidence)
                pattern.details = {
                    'golden_pairs_found': len(golden_pairs),
                    'average_ratio': np.mean([ratio for _, _, _, ratio in golden_pairs])
                }
                pattern.examples = golden_pairs[:5]
                pattern.frequency = len(golden_pairs)
                return pattern
        
        except Exception:
            pass
        
        return None
    
    def _detect_symmetry_pattern(self, df: pd.DataFrame) -> Optional[PatternResult]:
        """대칭 패턴 탐지"""
        try:
            number_cols = [f'num{i}' for i in range(1, 7)]
            if not all(col in df.columns for col in number_cols):
                return None
            
            symmetry_cases = []
            center = 23  # 1-45의 중심
            
            for idx, row in df.iterrows():
                numbers = [row[col] for col in number_cols if pd.notna(row[col])]
                if len(numbers) == 6:
                    # 중심축 대칭 확인
                    symmetric_pairs = []
                    used_numbers = set()
                    
                    for number in numbers:
                        if number not in used_numbers:
                            symmetric_counterpart = 2 * center - number
                            if symmetric_counterpart in numbers and symmetric_counterpart != number:
                                symmetric_pairs.append((number, symmetric_counterpart))
                                used_numbers.add(number)
                                used_numbers.add(symmetric_counterpart)
                    
                    if len(symmetric_pairs) >= 2:  # 2쌍 이상 대칭
                        symmetry_cases.append((idx, symmetric_pairs))
            
            if len(symmetry_cases) >= 3:
                confidence = len(symmetry_cases) / len(df)
                
                pattern = PatternResult('symmetry', confidence)
                pattern.details = {
                    'symmetric_draws': len(symmetry_cases),
                    'symmetry_rate': confidence
                }
                pattern.examples = symmetry_cases[:5]
                pattern.frequency = len(symmetry_cases)
                return pattern
        
        except Exception:
            pass
        
        return None
    
    def _detect_ratio_pattern(self, df: pd.DataFrame) -> Optional[PatternResult]:
        """특정 비율 패턴 탐지"""
        try:
            number_cols = [f'num{i}' for i in range(1, 7)]
            if not all(col in df.columns for col in number_cols):
                return None
            
            # 일반적인 비율들 (2:1, 3:2, 5:3 등)
            common_ratios = [2.0, 1.5, 1.67, 3.0, 2.5]
            ratio_matches = []
            
            for idx, row in df.iterrows():
                numbers = [row[col] for col in number_cols if pd.notna(row[col])]
                
                for i in range(len(numbers)):
                    for j in range(i+1, len(numbers)):
                        larger = max(numbers[i], numbers[j])
                        smaller = min(numbers[i], numbers[j])
                        
                        if smaller > 0:
                            ratio = larger / smaller
                            
                            # 일반적인 비율과 매치 확인
                            for target_ratio in common_ratios:
                                if abs(ratio - target_ratio) / target_ratio < 0.05:  # 5% 허용
                                    ratio_matches.append((idx, smaller, larger, ratio, target_ratio))
            
            if len(ratio_matches) >= 10:
                confidence = len(ratio_matches) / (len(df) * 15)
                
                pattern = PatternResult('common_ratios', confidence)
                pattern.details = {
                    'ratio_matches': len(ratio_matches),
                    'common_ratio_distribution': Counter([target for _, _, _, _, target in ratio_matches]).most_common()
                }
                pattern.examples = ratio_matches[:5]
                pattern.frequency = len(ratio_matches)
                return pattern
        
        except Exception:
            pass
        
        return None


class StatisticalPatternDetector:
    """통계적 패턴 탐지기"""
    
    def detect_patterns(self, df: pd.DataFrame) -> List[PatternResult]:
        """통계적 패턴들 탐지"""
        patterns = []
        
        # 분포 패턴
        distribution_pattern = self._detect_distribution_pattern(df)
        if distribution_pattern:
            patterns.append(distribution_pattern)
        
        # 상관관계 패턴
        correlation_pattern = self._detect_correlation_pattern(df)
        if correlation_pattern:
            patterns.append(correlation_pattern)
        
        # 클러스터 패턴
        cluster_pattern = self._detect_cluster_pattern(df)
        if cluster_pattern:
            patterns.append(cluster_pattern)
        
        # 트렌드 패턴
        trend_pattern = self._detect_trend_pattern(df)
        if trend_pattern:
            patterns.append(trend_pattern)
        
        return patterns
    
    def _detect_distribution_pattern(self, df: pd.DataFrame) -> Optional[PatternResult]:
        """분포 패턴 탐지"""
        try:
            number_cols = [f'num{i}' for i in range(1, 7)]
            if not all(col in df.columns for col in number_cols):
                return None
            
            # 모든 당첨번호 수집
            all_numbers = []
            for col in number_cols:
                all_numbers.extend(df[col].dropna().tolist())
            
            if len(all_numbers) < 50:
                return None
            
            # 분포 검정
            # 1. 정규성 검정
            _, p_normal = stats.normaltest(all_numbers)
            
            # 2. 균등 분포 검정
            observed_freq = [all_numbers.count(i) for i in range(1, 46)]
            expected_freq = len(all_numbers) / 45
            chi2_stat, p_uniform = stats.chisquare(observed_freq)
            
            # 3. 분포 특성 분석
            mean_num = np.mean(all_numbers)
            median_num = np.median(all_numbers)
            mode_num = statistics.mode(all_numbers)
            skewness = stats.skew(all_numbers)
            kurtosis = stats.kurtosis(all_numbers)
            
            # 패턴 강도 계산
            distribution_score = 0
            
            # 비정규성이 강하면 패턴 존재 가능성
            if p_normal < 0.05:
                distribution_score += 0.3
            
            # 비균등성이 강하면 패턴 존재
            if p_uniform < 0.05:
                distribution_score += 0.4
            
            # 왜도나 첨도가 크면 패턴 존재
            if abs(skewness) > 0.5 or abs(kurtosis) > 0.5:
                distribution_score += 0.3
            
            if distribution_score >= 0.3:
                pattern = PatternResult('distribution_anomaly', distribution_score)
                pattern.details = {
                    'normal_test_p': p_normal,
                    'uniform_test_p': p_uniform,
                    'mean': mean_num,
                    'median': median_num,
                    'mode': mode_num,
                    'skewness': skewness,
                    'kurtosis': kurtosis,
                    'distribution_type': self._classify_distribution(skewness, kurtosis)
                }
                return pattern
        
        except Exception:
            pass
        
        return None
    
    def _classify_distribution(self, skewness: float, kurtosis: float) -> str:
        """분포 유형 분류"""
        if abs(skewness) < 0.5 and abs(kurtosis) < 0.5:
            return 'approximately_normal'
        elif skewness > 0.5:
            return 'right_skewed'
        elif skewness < -0.5:
            return 'left_skewed'
        elif kurtosis > 0.5:
            return 'heavy_tailed'
        else:
            return 'light_tailed'
    
    def _detect_correlation_pattern(self, df: pd.DataFrame) -> Optional[PatternResult]:
        """상관관계 패턴 탐지"""
        try:
            number_cols = [f'num{i}' for i in range(1, 7)]
            if not all(col in df.columns for col in number_cols):
                return None
            
            # 번호 간 상관관계 계산
            corr_matrix = df[number_cols].corr()
            
            # 대각선 제외하고 상관관계 분석
            significant_correlations = []
            
            for i in range(len(number_cols)):
                for j in range(i+1, len(number_cols)):
                    corr_value = corr_matrix.iloc[i, j]
                    if not pd.isna(corr_value) and abs(corr_value) > 0.1:  # 10% 이상 상관관계
                        significant_correlations.append({
                            'col1': number_cols[i],
                            'col2': number_cols[j],
                            'correlation': corr_value,
                            'strength': 'strong' if abs(corr_value) > 0.3 else 'moderate'
                        })
            
            if significant_correlations:
                avg_correlation = np.mean([abs(c['correlation']) for c in significant_correlations])
                
                pattern = PatternResult('number_correlation', avg_correlation)
                pattern.details = {
                    'significant_correlations': len(significant_correlations),
                    'correlations': significant_correlations,
                    'average_correlation': avg_correlation
                }
                return pattern
        
        except Exception:
            pass
        
        return None
    
    def _detect_cluster_pattern(self, df: pd.DataFrame) -> Optional[PatternResult]:
        """클러스터 패턴 탐지"""
        try:
            number_cols = [f'num{i}' for i in range(1, 7)]
            if not all(col in df.columns for col in number_cols):
                return None
            
            # 번호들의 공간적 클러스터링 분석
            cluster_cases = []
            
            for idx, row in df.iterrows():
                numbers = [row[col] for col in number_cols if pd.notna(row[col])]
                if len(numbers) == 6:
                    sorted_numbers = sorted(numbers)
                    
                    # 클러스터 감지 (연속된 번호들의 그룹)
                    clusters = self._find_clusters(sorted_numbers)
                    
                    if len(clusters) <= 3:  # 3개 이하 클러스터 (집중됨)
                        cluster_cases.append((idx, clusters))
            
            if len(cluster_cases) >= len(df) * 0.2:  # 20% 이상에서 클러스터링
                confidence = len(cluster_cases) / len(df)
                
                pattern = PatternResult('number_clustering', confidence)
                pattern.details = {
                    'clustered_draws': len(cluster_cases),
                    'clustering_rate': confidence,
                    'average_clusters_per_draw': np.mean([len(clusters) for _, clusters in cluster_cases])
                }
                pattern.examples = cluster_cases[:5]
                pattern.frequency = len(cluster_cases)
                return pattern
        
        except Exception:
            pass
        
        return None
    
    def _find_clusters(self, sorted_numbers: List[int], max_gap: int = 5) -> List[List[int]]:
        """번호들을 클러스터로 그룹화"""
        if not sorted_numbers:
            return []
        
        clusters = []
        current_cluster = [sorted_numbers[0]]
        
        for i in range(1, len(sorted_numbers)):
            if sorted_numbers[i] - sorted_numbers[i-1] <= max_gap:
                current_cluster.append(sorted_numbers[i])
            else:
                clusters.append(current_cluster)
                current_cluster = [sorted_numbers[i]]
        
        clusters.append(current_cluster)
        return clusters
    
    def _detect_trend_pattern(self, df: pd.DataFrame) -> Optional[PatternResult]:
        """트렌드 패턴 탐지"""
        try:
            number_cols = [f'num{i}' for i in range(1, 7)]
            if not all(col in df.columns for col in number_cols):
                return None
            
            # 시간에 따른 번호 크기 트렌드 분석
            if len(df) < 20:
                return None
            
            # 각 회차별 평균 번호 계산
            draw_averages = []
            for idx, row in df.iterrows():
                numbers = [row[col] for col in number_cols if pd.notna(row[col])]
                if numbers:
                    draw_averages.append(np.mean(numbers))
            
            if len(draw_averages) >= 20:
                # 선형 트렌드 검정
                x = np.arange(len(draw_averages))
                slope, intercept, r_value, p_value, std_err = stats.linregress(x, draw_averages)
                
                # 유의한 트렌드인지 확인
                if p_value < 0.05 and abs(r_value) > 0.2:
                    trend_direction = 'increasing' if slope > 0 else 'decreasing'
                    
                    pattern = PatternResult('number_trend', abs(r_value))
                    pattern.details = {
                        'trend_direction': trend_direction,
                        'slope': slope,
                        'correlation': r_value,
                        'p_value': p_value,
                        'trend_strength': abs(slope) * len(draw_averages)
                    }
                    pattern.trend = trend_direction
                    return pattern
        
        except Exception:
            pass
        
        return None


# === 모듈 테스트 ===
if __name__ == "__main__":
    # 테스트용 데이터 생성
    np.random.seed(42)
    
    # 패턴이 있는 테스트 데이터 생성
    test_data = []
    
    for i in range(100):
        # 일부 패턴을 의도적으로 삽입
        if i % 10 == 0:  # 연속 번호 패턴
            numbers = [1, 2, 3, 15, 25, 35]
        elif i % 15 == 0:  # 등차수열 패턴
            start = np.random.randint(1, 20)
            numbers = [start, start+5, start+10, start+15, start+20, start+25]
            numbers = [n for n in numbers if n <= 45][:6]
            while len(numbers) < 6:
                numbers.append(np.random.randint(1, 46))
        elif i % 20 == 0:  # 피보나치 패턴
            fib_nums = [1, 2, 3, 5, 8, 13, 21, 34]
            numbers = np.random.choice(fib_nums, size=6, replace=False).tolist()
        else:  # 일반적인 랜덤 패턴
            numbers = sorted(np.random.choice(range(1, 46), size=6, replace=False))
        
        # 데이터 행 생성
        row = {
            'round': 1000 + i,
            'draw_date': datetime(2020, 1, 1) + timedelta(days=i*3),
        }
        
        for j, num in enumerate(numbers[:6], 1):
            row[f'num{j}'] = num
        
        test_data.append(row)
    
    test_df = pd.DataFrame(test_data)
    
    print("=== 패턴 탐지기 테스트 ===")
    print(f"테스트 데이터: {len(test_df)}행")
    print(f"샘플 데이터:\n{test_df.head()}")
    
    try:
        # 패턴 탐지기 초기화
        detector = LottoPatternDetector(min_confidence=0.05)
        
        # 전체 패턴 탐지
        all_patterns = detector.detect_all_patterns(test_df, include_weak=True)
        
        print(f"\n🔍 패턴 탐지 결과:")
        for pattern_type, patterns in all_patterns.items():
            print(f"\n{pattern_type.upper()} 패턴:")
            if patterns:
                for i, pattern in enumerate(patterns[:3], 1):  # 상위 3개만 표시
                    print(f"  {i}. {pattern.pattern_type}")
                    print(f"     신뢰도: {pattern.confidence:.3f}")
                    print(f"     빈도: {pattern.frequency}")
                    if pattern.details:
                        for key, value in list(pattern.details.items())[:2]:  # 주요 정보만
                            print(f"     {key}: {value}")
            else:
                print("  패턴 없음")
        
        # 패턴 요약
        summary = detector.get_pattern_summary(all_patterns)
        print(f"\n📊 패턴 요약:")
        print(f"  총 패턴 수: {summary['total_patterns']}")
        print(f"  최고 신뢰도: {summary['highest_confidence']:.3f}")
        print(f"  강한 패턴 수: {summary['strong_patterns']}")
        print(f"  주요 패턴 타입: {summary['most_frequent_type']}")
        
        if summary['recommendations']:
            print(f"\n💡 권장사항:")
            for rec in summary['recommendations']:
                print(f"  - {rec}")
        
        # 개별 탐지기 테스트
        print(f"\n🧪 개별 탐지기 테스트:")
        
        # 순차 패턴 탐지기
        seq_patterns = detector.sequential_detector.detect_patterns(test_df)
        print(f"  순차 패턴: {len(seq_patterns)}개")
        
        # 주기 패턴 탐지기
        cyc_patterns = detector.cyclic_detector.detect_patterns(test_df)
        print(f"  주기 패턴: {len(cyc_patterns)}개")
        
        # 기하학적 패턴 탐지기
        geo_patterns = detector.geometric_detector.detect_patterns(test_df)
        print(f"  기하학적 패턴: {len(geo_patterns)}개")
        
        # 통계적 패턴 탐지기
        stat_patterns = detector.statistical_detector.detect_patterns(test_df)
        print(f"  통계적 패턴: {len(stat_patterns)}개")
        
        print("\n✅ 패턴 탐지기 테스트 완료")
        
    except Exception as e:
        print(f"❌ 테스트 중 오류: {e}")
        import traceback
        traceback.print_exc()
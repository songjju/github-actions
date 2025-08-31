"""
파일명: src/genius_formulas/formula_01_genius_insight.py
목적: 천재적 통찰 도출 공식 (GI = (O × C × P × S) / (A + B)) 구현
작성자: AI Assistant
작성일: 2025-08-30
버전: 1.0

천재적 통찰 공식:
GI = (O × C × P × S) / (A + B)

컴포넌트:
- O (Observation): 관찰의 깊이 (1-10)
- C (Connection): 연결의 독창성 (1-10)
- P (Pattern): 패턴 인식 능력 (1-10)
- S (Synthesis): 종합적 사고 (1-10)
- A (Assumption): 고정관념 수준 (1-10) - 낮을수록 좋음
- B (Bias): 편향 정도 (1-10) - 낮을수록 좋음
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any, Optional
import math
import random
from collections import Counter, defaultdict
import logging
from pathlib import Path
import sys

# 상위 디렉토리의 config 임포트
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.utils.config import Config, LottoConstants, GeniusFormulaConstants

class GeniusInsightFormula:
    """천재적 통찰 공식 구현 클래스"""
    
    def __init__(self, lotto_data: pd.DataFrame, statistics: Dict[str, Any]):
        """
        초기화
        
        Args:
            lotto_data (pd.DataFrame): 로또 데이터
            statistics (Dict[str, Any]): 기초 통계 분석 결과
        """
        self.data = lotto_data.copy()
        self.stats = statistics
        self.logger = self._setup_logger()
        
        # 컴포넌트 점수 저장
        self.component_scores = {}
        self.number_insights = {}  # 각 번호별 통찰 점수
        
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
    
    def calculate_genius_insight(self, target_numbers: Optional[List[int]] = None) -> Dict[str, Any]:
        """
        천재적 통찰 공식 계산
        
        Args:
            target_numbers (Optional[List[int]]): 특정 번호들에 대한 분석 (None이면 전체)
            
        Returns:
            Dict[str, Any]: 통찰 분석 결과
        """
        self.logger.info("천재적 통찰 공식 계산 시작")
        
        try:
            # 각 컴포넌트 계산
            observation_scores = self._calculate_observation_depth()
            connection_scores = self._calculate_connection_creativity()
            pattern_scores = self._calculate_pattern_recognition()
            synthesis_scores = self._calculate_synthetic_thinking()
            assumption_levels = self._calculate_assumption_level()
            bias_degrees = self._calculate_bias_degree()
            
            # 번호별 GI 점수 계산
            gi_scores = {}
            for number in range(1, LottoConstants.MAX_NUMBER + 1):
                O = observation_scores.get(number, 5.0)
                C = connection_scores.get(number, 5.0)
                P = pattern_scores.get(number, 5.0)
                S = synthesis_scores.get(number, 5.0)
                A = assumption_levels.get(number, 5.0)
                B = bias_degrees.get(number, 5.0)
                
                # GI 공식 적용
                denominator = max(A + B, 1.0)  # 0으로 나누기 방지
                gi_score = (O * C * P * S) / denominator
                gi_scores[number] = gi_score
            
            # 결과 저장
            self.component_scores = {
                'observation': observation_scores,
                'connection': connection_scores,
                'pattern': pattern_scores,
                'synthesis': synthesis_scores,
                'assumption': assumption_levels,
                'bias': bias_degrees
            }
            
            self.number_insights = gi_scores
            
            # 상위 번호들 선정
            top_numbers = self._select_top_insightful_numbers(gi_scores)
            
            # 추가 분석
            insights_analysis = self._analyze_insights(gi_scores)
            
            result = {
                'gi_scores': gi_scores,
                'component_scores': self.component_scores,
                'top_insightful_numbers': top_numbers,
                'insights_analysis': insights_analysis,
                'formula_summary': self._generate_formula_summary(gi_scores)
            }
            
            self.logger.info("천재적 통찰 공식 계산 완료")
            return result
            
        except Exception as e:
            self.logger.error(f"천재적 통찰 계산 중 오류: {e}")
            raise
    
    def _calculate_observation_depth(self) -> Dict[int, float]:
        """관찰의 깊이 계산 (O 컴포넌트)"""
        observation_scores = {}
        
        # 기초 통계에서 빈도 정보 가져오기
        frequency_data = self.stats.get('frequency_analysis', {})
        frequency_count = frequency_data.get('frequency_count', {})
        
        if not frequency_count:
            # 기본값 반환
            return {i: 5.0 for i in range(1, 46)}
        
        # 빈도 기반 관찰 점수
        max_freq = max(frequency_count.values()) if frequency_count else 1
        min_freq = min(frequency_count.values()) if frequency_count else 1
        freq_range = max_freq - min_freq
        
        for number in range(1, LottoConstants.MAX_NUMBER + 1):
            base_score = 5.0  # 기본 점수
            
            # 1. 빈도 기반 관찰
            freq = frequency_count.get(number, 0)
            if freq_range > 0:
                freq_normalized = (freq - min_freq) / freq_range
                freq_score = 1 + freq_normalized * 8  # 1-9 범위로 변환
            else:
                freq_score = 5.0
            
            # 2. 최근 출현 패턴 관찰
            recent_score = self._observe_recent_patterns(number)
            
            # 3. 주기적 패턴 관찰
            cycle_score = self._observe_cyclical_patterns(number)
            
            # 4. 구간별 특성 관찰
            section_score = self._observe_section_characteristics(number)
            
            # 종합 관찰 점수 (가중 평균)
            observation_score = (
                freq_score * 0.3 +
                recent_score * 0.3 +
                cycle_score * 0.2 +
                section_score * 0.2
            )
            
            # 1-10 범위로 제한
            observation_scores[number] = max(1.0, min(10.0, observation_score))
        
        return observation_scores
    
    def _observe_recent_patterns(self, number: int) -> float:
        """최근 패턴 관찰"""
        recent_appearances = []
        
        for idx, row in self.data.iterrows():
            if number in row['winning_numbers']:
                recent_appearances.append(idx)
        
        if len(recent_appearances) < 2:
            return 5.0  # 충분한 데이터가 없으면 중간값
        
        # 최근 출현 간격 분석
        intervals = []
        for i in range(1, len(recent_appearances)):
            interval = recent_appearances[i] - recent_appearances[i-1]
            intervals.append(interval)
        
        if not intervals:
            return 5.0
        
        # 간격의 규칙성 점수 (변동계수 기반)
        mean_interval = np.mean(intervals)
        std_interval = np.std(intervals)
        
        if mean_interval == 0:
            return 5.0
        
        coefficient_of_variation = std_interval / mean_interval
        regularity_score = max(1.0, 10.0 - coefficient_of_variation * 5)
        
        return min(10.0, regularity_score)
    
    def _observe_cyclical_patterns(self, number: int) -> float:
        """주기적 패턴 관찰"""
        appearances = []
        
        for idx, row in self.data.iterrows():
            if number in row['winning_numbers']:
                appearances.append(idx)
        
        if len(appearances) < 3:
            return 5.0
        
        # 간단한 주기성 탐지 (연속 간격의 패턴)
        intervals = []
        for i in range(1, len(appearances)):
            interval = appearances[i] - appearances[i-1]
            intervals.append(interval)
        
        # 주기성 점수 (간격의 반복 패턴 탐지)
        interval_counter = Counter(intervals)
        most_common_interval = interval_counter.most_common(1)[0] if interval_counter else (0, 0)
        
        # 가장 흔한 간격의 비율로 주기성 점수 계산
        if len(intervals) > 0:
            cyclical_score = (most_common_interval[1] / len(intervals)) * 10
        else:
            cyclical_score = 5.0
        
        return max(1.0, min(10.0, cyclical_score))
    
    def _observe_section_characteristics(self, number: int) -> float:
        """구간별 특성 관찰"""
        # 번호가 속한 구간의 특성 분석
        if number <= 15:
            section = 'low'
        elif number <= 30:
            section = 'mid'
        else:
            section = 'high'
        
        # 해당 구간의 출현 빈도 특성
        section_data = self.stats.get('distribution_analysis', {}).get('section_distribution', {})
        section_count = section_data.get(section, 0)
        total_section_count = sum(section_data.values()) if section_data else 1
        
        if total_section_count > 0:
            section_ratio = section_count / total_section_count
            section_score = section_ratio * 10
        else:
            section_score = 5.0
        
        return max(1.0, min(10.0, section_score))
    
    def _calculate_connection_creativity(self) -> Dict[int, float]:
        """연결의 독창성 계산 (C 컴포넌트)"""
        connection_scores = {}
        
        for number in range(1, LottoConstants.MAX_NUMBER + 1):
            # 1. 수학적 특성과의 연결
            math_score = self._connect_mathematical_properties(number)
            
            # 2. 사회적/문화적 연결
            cultural_score = self._connect_cultural_significance(number)
            
            # 3. 자연계 패턴과의 연결
            nature_score = self._connect_nature_patterns(number)
            
            # 4. 다른 번호들과의 관계
            relational_score = self._connect_number_relationships(number)
            
            # 종합 연결 점수
            connection_score = (
                math_score * 0.3 +
                cultural_score * 0.2 +
                nature_score * 0.2 +
                relational_score * 0.3
            )
            
            connection_scores[number] = max(1.0, min(10.0, connection_score))
        
        return connection_scores
    
    def _connect_mathematical_properties(self, number: int) -> float:
        """수학적 특성 연결"""
        score = 5.0  # 기본 점수
        
        # 소수 여부
        if self._is_prime(number):
            score += 1.5
        
        # 완전제곱수 여부
        if int(math.sqrt(number))**2 == number:
            score += 1.0
        
        # 피보나치 수열 여부
        if self._is_fibonacci(number):
            score += 2.0
        
        # 팰린드롬 여부 (두 자리수에서)
        if number >= 10 and str(number) == str(number)[::-1]:
            score += 1.0
        
        # 특별한 수학적 의미 (7, 13, 21 등)
        special_numbers = {7: 1.5, 13: 1.0, 21: 1.0, 42: 1.5}  # 의미있는 숫자들
        if number in special_numbers:
            score += special_numbers[number]
        
        return min(10.0, score)
    
    def _connect_cultural_significance(self, number: int) -> float:
        """문화적 의미 연결"""
        score = 5.0
        
        # 한국 문화에서의 특별한 의미
        lucky_numbers_kr = {3: 1.0, 7: 1.5, 8: 2.0, 9: 1.0}  # 한국에서 선호하는 숫자
        unlucky_numbers_kr = {4: -1.0}  # 기피하는 숫자
        
        if number in lucky_numbers_kr:
            score += lucky_numbers_kr[number]
        
        if number in unlucky_numbers_kr:
            score += unlucky_numbers_kr[number]
        
        # 생년월일과 관련된 범위 (1-31)
        if 1 <= number <= 31:
            score += 0.5
        
        # 특별한 날짜나 기념일과 관련
        special_dates = {1: 0.5, 12: 0.5, 25: 0.5, 31: 0.5}
        if number in special_dates:
            score += special_dates[number]
        
        return max(1.0, min(10.0, score))
    
    def _connect_nature_patterns(self, number: int) -> float:
        """자연계 패턴 연결"""
        score = 5.0
        
        # 황금비와의 관련성
        golden_ratio = 1.618
        if abs(number - golden_ratio * 10) < 2:  # 16.18 근처
            score += 1.5
        
        # 원주율과의 관련성
        pi_digits = [3, 1, 4, 1, 5, 9, 2, 6, 5, 3, 5, 8, 9, 7, 9]
        if number in pi_digits:
            score += 1.0
        
        # 자연수의 조화 (5, 10, 15, 20 등 5의 배수)
        if number % 5 == 0:
            score += 0.5
        
        # 이진법에서의 특별함 (2의 거듭제곱)
        powers_of_2 = [1, 2, 4, 8, 16, 32]
        if number in powers_of_2:
            score += 1.0
        
        return min(10.0, score)
    
    def _connect_number_relationships(self, number: int) -> float:
        """다른 번호들과의 관계"""
        # 동시 출현 빈도 기반 점수
        cooccurrence_data = self.stats.get('correlation_analysis', {}).get('number_cooccurrence', {})
        top_companions = cooccurrence_data.get('top_companions', {})
        
        if number in top_companions:
            companions = top_companions[number]
            # 동반 출현이 많을수록 높은 점수
            companion_score = min(10.0, 5.0 + len(companions) * 0.5)
        else:
            companion_score = 5.0
        
        return companion_score
    
    def _calculate_pattern_recognition(self) -> Dict[int, float]:
        """패턴 인식 능력 계산 (P 컴포넌트)"""
        pattern_scores = {}
        
        for number in range(1, LottoConstants.MAX_NUMBER + 1):
            # 1. 시계열 패턴 인식
            temporal_score = self._recognize_temporal_patterns(number)
            
            # 2. 빈도 패턴 인식
            frequency_score = self._recognize_frequency_patterns(number)
            
            # 3. 순환 패턴 인식
            cyclical_score = self._recognize_cyclical_patterns(number)
            
            # 4. 상관 패턴 인식
            correlation_score = self._recognize_correlation_patterns(number)
            
            # 종합 패턴 점수
            pattern_score = (
                temporal_score * 0.3 +
                frequency_score * 0.3 +
                cyclical_score * 0.2 +
                correlation_score * 0.2
            )
            
            pattern_scores[number] = max(1.0, min(10.0, pattern_score))
        
        return pattern_scores
    
    def _recognize_temporal_patterns(self, number: int) -> float:
        """시계열 패턴 인식"""
        # 번호의 시간에 따른 출현 패턴 분석
        appearances = []
        for idx, row in self.data.iterrows():
            if number in row['winning_numbers']:
                appearances.append(len(self.data) - idx)  # 시간 순서로 변환
        
        if len(appearances) < 3:
            return 5.0
        
        # 트렌드 분석 (최근에 더 자주 나오는지)
        recent_count = sum(1 for app in appearances if app <= 52)  # 최근 1년
        older_count = len(appearances) - recent_count
        
        if older_count > 0:
            trend_ratio = recent_count / older_count
            trend_score = 5.0 + (trend_ratio - 1) * 2  # 트렌드에 따른 점수
        else:
            trend_score = 5.0
        
        return max(1.0, min(10.0, trend_score))
    
    def _recognize_frequency_patterns(self, number: int) -> float:
        """빈도 패턴 인식"""
        frequency_data = self.stats.get('frequency_analysis', {})
        frequency_count = frequency_data.get('frequency_count', {})
        
        if not frequency_count:
            return 5.0
        
        freq = frequency_count.get(number, 0)
        avg_freq = np.mean(list(frequency_count.values()))
        
        # 평균에서의 편차를 점수로 변환
        if avg_freq > 0:
            deviation_ratio = freq / avg_freq
            frequency_score = 5.0 + (deviation_ratio - 1) * 3
        else:
            frequency_score = 5.0
        
        return max(1.0, min(10.0, frequency_score))
    
    def _recognize_cyclical_patterns(self, number: int) -> float:
        """순환 패턴 인식"""
        # 이전에 구현한 cyclical pattern 로직 재사용
        return self._observe_cyclical_patterns(number)
    
    def _recognize_correlation_patterns(self, number: int) -> float:
        """상관 패턴 인식"""
        # 다른 번호들과의 상관관계 패턴
        cooccurrence_data = self.stats.get('correlation_analysis', {}).get('number_cooccurrence', {})
        
        if not cooccurrence_data:
            return 5.0
        
        # 이 번호와 함께 자주 나오는 번호들의 패턴
        top_companions = cooccurrence_data.get('top_companions', {}).get(number, [])
        
        if top_companions:
            # 상위 동반 번호들과의 관계 강도
            total_cooccurrence = sum(count for _, count in top_companions)
            correlation_score = min(10.0, 5.0 + total_cooccurrence / 50)
        else:
            correlation_score = 5.0
        
        return correlation_score
    
    def _calculate_synthetic_thinking(self) -> Dict[int, float]:
        """종합적 사고 계산 (S 컴포넌트)"""
        synthesis_scores = {}
        
        for number in range(1, LottoConstants.MAX_NUMBER + 1):
            # 모든 이전 분석 결과들을 종합
            
            # 1. 다차원적 통합
            multi_dimensional_score = self._synthesize_multi_dimensional(number)
            
            # 2. 메타 패턴 통합
            meta_pattern_score = self._synthesize_meta_patterns(number)
            
            # 3. 직관적 통합
            intuitive_score = self._synthesize_intuitive_insights(number)
            
            # 4. 전체적 맥락 통합
            contextual_score = self._synthesize_contextual_factors(number)
            
            # 종합 점수
            synthesis_score = (
                multi_dimensional_score * 0.3 +
                meta_pattern_score * 0.3 +
                intuitive_score * 0.2 +
                contextual_score * 0.2
            )
            
            synthesis_scores[number] = max(1.0, min(10.0, synthesis_score))
        
        return synthesis_scores
    
    def _synthesize_multi_dimensional(self, number: int) -> float:
        """다차원적 통합"""
        # 시간, 빈도, 패턴, 관계 등 모든 차원의 정보 통합
        dimensions = [
            self._observe_recent_patterns(number),
            self._recognize_frequency_patterns(number),
            self._connect_number_relationships(number),
            self._connect_mathematical_properties(number)
        ]
        
        # 차원들의 조화로운 통합 (분산이 낮을수록 좋음)
        mean_score = np.mean(dimensions)
        variance = np.var(dimensions)
        
        # 낮은 분산은 일관성을 의미하므로 가점
        consistency_bonus = max(0, 2 - variance)
        
        return min(10.0, mean_score + consistency_bonus)
    
    def _synthesize_meta_patterns(self, number: int) -> float:
        """메타 패턴 통합"""
        # 패턴들 간의 패턴 (메타 패턴) 탐지
        base_score = 5.0
        
        # 번호의 다양한 특성들이 일치하는 방향을 가리키는지 확인
        characteristics = {
            'is_prime': self._is_prime(number),
            'is_even': number % 2 == 0,
            'is_low_section': number <= 15,
            'is_fibonacci': self._is_fibonacci(number),
            'is_frequent': self._is_frequent_number(number)
        }
        
        # 특성들의 일관성 평가
        positive_traits = sum(characteristics.values())
        trait_consistency = abs(positive_traits - 2.5) / 2.5  # 2.5에서의 편차
        
        meta_score = base_score + (1 - trait_consistency) * 3
        
        return min(10.0, meta_score)
    
    def _synthesize_intuitive_insights(self, number: int) -> float:
        """직관적 통합"""
        # 데이터를 넘어선 직관적 평가
        intuitive_factors = []
        
        # 1. 시각적 매력 (번호의 형태)
        visual_score = 5.0
        if str(number).count('1') > 0:  # 1이 포함된 번호
            visual_score += 0.5
        if len(set(str(number))) == len(str(number)):  # 중복 숫자 없음
            visual_score += 0.5
        intuitive_factors.append(visual_score)
        
        # 2. 음성학적 매력 (발음의 용이성)
        phonetic_score = 5.0
        if number in [1, 2, 3, 5, 7, 8, 9, 10]:  # 발음하기 쉬운 숫자
            phonetic_score += 1.0
        intuitive_factors.append(phonetic_score)
        
        # 3. 심리적 선호도
        psychological_score = 5.0
        if number in [7, 11, 13, 21, 33]:  # 심리적으로 선호되는 숫자
            psychological_score += 1.5
        intuitive_factors.append(psychological_score)
        
        return min(10.0, np.mean(intuitive_factors))
    
    def _synthesize_contextual_factors(self, number: int) -> float:
        """전체적 맥락 통합"""
        # 현재 상황과 맥락을 고려한 통합
        contextual_score = 5.0
        
        # 1. 시기적 적절성 (계절, 월 등)
        # 현재 구현에서는 단순화
        seasonal_score = 5.0 + random.uniform(-1, 1)  # 임의의 계절적 요소
        
        # 2. 시스템 전체의 균형
        balance_score = self._evaluate_system_balance(number)
        
        # 3. 예측의 다양성 고려
        diversity_score = self._evaluate_prediction_diversity(number)
        
        contextual_score = (seasonal_score + balance_score + diversity_score) / 3
        
        return max(1.0, min(10.0, contextual_score))
    
    def _calculate_assumption_level(self) -> Dict[int, float]:
        """고정관념 수준 계산 (A 컴포넌트) - 낮을수록 좋음"""
        assumption_levels = {}
        
        for number in range(1, LottoConstants.MAX_NUMBER + 1):
            # 기본 가정 수준
            base_assumption = 5.0
            
            # 1. 일반적 선호도로 인한 고정관념
            if number in [7, 8, 9, 11, 13]:  # 일반적으로 선호하는 숫자
                base_assumption += 2.0  # 고정관념 증가
            
            # 2. 회피 숫자로 인한 고정관념
            if number == 4:  # 기피 숫자
                base_assumption += 1.5
            
            # 3. 생일 관련 편향
            if 1 <= number <= 31:  # 생일 범위
                base_assumption += 1.0
            
            # 4. 패턴에 대한 고정관념 (너무 규칙적인 것에 대한 편견)
            if number % 5 == 0:  # 5의 배수
                base_assumption += 0.5
            
            # 5. 극값에 대한 편견
            if number <= 5 or number >= 40:
                base_assumption += 0.5
            
            assumption_levels[number] = max(1.0, min(10.0, base_assumption))
        
        return assumption_levels
    
    def _calculate_bias_degree(self) -> Dict[int, float]:
        """편향 정도 계산 (B 컴포넌트) - 낮을수록 좋음"""
        bias_degrees = {}
        
        for number in range(1, LottoConstants.MAX_NUMBER + 1):
            base_bias = 5.0
            
            # 1. 빈도 편향 (너무 자주/드물게 나온 번호에 대한 편향)
            frequency_data = self.stats.get('frequency_analysis', {})
            if frequency_data:
                hot_numbers = frequency_data.get('hot_numbers', [])
                cold_numbers = frequency_data.get('cold_numbers', [])
                
                if number in hot_numbers:
                    base_bias += 1.5  # 뜨거운 번호에 대한 편향
                if number in cold_numbers:
                    base_bias += 1.0  # 차가운 번호에 대한 편향
            
            # 2. 최근성 편향
            recent_appearance = self._get_recent_appearance(number)
            if recent_appearance is not None and recent_appearance < 10:
                base_bias += 1.0  # 최근에 나온 번호에 대한 편향
            
            # 3. 대표성 휴리스틱 편향
            if self._looks_random(number):
                base_bias -= 0.5  # 무작위처럼 보이는 번호는 편향 감소
            else:
                base_bias += 0.5  # 패턴이 있어 보이는 번호는 편향 증가
            
            # 4. 확증 편향 (기존 믿음 강화)
            if self._confirms_existing_beliefs(number):
                base_bias += 1.0
            
            # 5. 가용성 휴리스틱 (기억하기 쉬운 번호)
            if self._is_memorable(number):
                base_bias += 0.5
            
            bias_degrees[number] = max(1.0, min(10.0, base_bias))
        
        return bias_degrees
    
    def _select_top_insightful_numbers(self, gi_scores: Dict[int, float], count: int = 15) -> List[Tuple[int, float]]:
        """상위 통찰력 있는 번호들 선정"""
        # GI 점수를 기준으로 정렬
        sorted_numbers = sorted(gi_scores.items(), key=lambda x: x[1], reverse=True)
        
        # 상위 번호들 선정 (다양성도 고려)
        top_numbers = []
        selected_sections = {'low': 0, 'mid': 0, 'high': 0}
        
        for number, score in sorted_numbers:
            if len(top_numbers) >= count:
                break
            
            # 구간 다양성 확인
            section = self._get_number_section(number)
            if selected_sections[section] >= count // 3 + 2:  # 각 구간 최대 개수 제한
                continue
            
            top_numbers.append((number, score))
            selected_sections[section] += 1
        
        return top_numbers
    
    def _analyze_insights(self, gi_scores: Dict[int, float]) -> Dict[str, Any]:
        """통찰 분석"""
        scores = list(gi_scores.values())
        
        return {
            'score_statistics': {
                'mean': np.mean(scores),
                'std': np.std(scores),
                'min': min(scores),
                'max': max(scores),
                'median': np.median(scores)
            },
            'score_distribution': {
                'high_insight': len([s for s in scores if s >= 7.0]),
                'medium_insight': len([s for s in scores if 4.0 <= s < 7.0]),
                'low_insight': len([s for s in scores if s < 4.0])
            },
            'section_analysis': self._analyze_section_insights(gi_scores),
            'pattern_insights': self._extract_pattern_insights(gi_scores)
        }
    
    def _analyze_section_insights(self, gi_scores: Dict[int, float]) -> Dict[str, Any]:
        """구간별 통찰 분석"""
        sections = {'low': [], 'mid': [], 'high': []}
        
        for number, score in gi_scores.items():
            section = self._get_number_section(number)
            sections[section].append(score)
        
        section_stats = {}
        for section, scores in sections.items():
            if scores:
                section_stats[section] = {
                    'mean': np.mean(scores),
                    'count': len(scores),
                    'top_numbers': [num for num, score in gi_scores.items() 
                                   if self._get_number_section(num) == section and score >= np.mean(scores)]
                }
        
        return section_stats
    
    def _extract_pattern_insights(self, gi_scores: Dict[int, float]) -> List[str]:
        """패턴 통찰 추출"""
        insights = []
        
        # 높은 점수를 받은 번호들의 특성 분석
        high_score_numbers = [num for num, score in gi_scores.items() if score >= 7.0]
        
        if high_score_numbers:
            # 소수 비율
            prime_count = sum(1 for num in high_score_numbers if self._is_prime(num))
            if prime_count / len(high_score_numbers) > 0.4:
                insights.append("소수들이 높은 통찰 점수를 보입니다.")
            
            # 홀짝 비율
            odd_count = sum(1 for num in high_score_numbers if num % 2 == 1)
            if odd_count / len(high_score_numbers) > 0.7:
                insights.append("홀수가 통찰적으로 우세합니다.")
            elif odd_count / len(high_score_numbers) < 0.3:
                insights.append("짝수가 통찰적으로 우세합니다.")
            
            # 구간 분포
            low_count = sum(1 for num in high_score_numbers if num <= 15)
            mid_count = sum(1 for num in high_score_numbers if 16 <= num <= 30)
            high_count = sum(1 for num in high_score_numbers if num >= 31)
            
            total = len(high_score_numbers)
            if low_count / total > 0.5:
                insights.append("저구간(1-15)에서 높은 통찰을 보입니다.")
            elif high_count / total > 0.5:
                insights.append("고구간(31-45)에서 높은 통찰을 보입니다.")
            elif mid_count / total > 0.5:
                insights.append("중구간(16-30)에서 높은 통찰을 보입니다.")
        
        return insights
    
    def _generate_formula_summary(self, gi_scores: Dict[int, float]) -> Dict[str, Any]:
        """공식 요약 생성"""
        return {
            'formula': 'GI = (O × C × P × S) / (A + B)',
            'component_weights': {
                'observation': '관찰의 깊이',
                'connection': '연결의 독창성',
                'pattern': '패턴 인식 능력',
                'synthesis': '종합적 사고',
                'assumption': '고정관념 수준 (낮을수록 좋음)',
                'bias': '편향 정도 (낮을수록 좋음)'
            },
            'top_5_numbers': sorted(gi_scores.items(), key=lambda x: x[1], reverse=True)[:5],
            'insights_summary': f"총 {len(gi_scores)}개 번호 중 평균 GI 점수: {np.mean(list(gi_scores.values())):.2f}"
        }
    
    def predict_numbers(self, count: int = 6) -> List[int]:
        """
        천재적 통찰을 기반으로 번호 예측
        
        Args:
            count (int): 예측할 번호 개수
            
        Returns:
            List[int]: 예측된 번호 리스트
        """
        # GI 점수 계산 (아직 계산되지 않았다면)
        if not self.number_insights:
            self.calculate_genius_insight()
        
        # GI 점수를 확률로 변환
        scores = list(self.number_insights.values())
        min_score = min(scores)
        max_score = max(scores)
        score_range = max_score - min_score
        
        if score_range == 0:
            # 모든 점수가 같다면 균등 확률
            probabilities = [1/45] * 45
        else:
            # 점수를 0-1 범위로 정규화 후 확률로 변환
            probabilities = []
            for number in range(1, 46):
                normalized_score = (self.number_insights[number] - min_score) / score_range
                # 지수 함수를 사용해 높은 점수에 더 높은 확률 부여
                probability = math.exp(normalized_score * 2)
                probabilities.append(probability)
        
        # 확률 정규화
        total_prob = sum(probabilities)
        probabilities = [p / total_prob for p in probabilities]
        
        # 가중 랜덤 선택
        selected_numbers = []
        available_numbers = list(range(1, 46))
        available_probs = probabilities.copy()
        
        for _ in range(count):
            if not available_numbers:
                break
            
            # 누적 확률을 사용한 선택
            cumulative_probs = []
            cumsum = 0
            for prob in available_probs:
                cumsum += prob
                cumulative_probs.append(cumsum)
            
            rand = random.random() * cumsum
            selected_idx = 0
            for i, cum_prob in enumerate(cumulative_probs):
                if rand <= cum_prob:
                    selected_idx = i
                    break
            
            selected_number = available_numbers[selected_idx]
            selected_numbers.append(selected_number)
            
            # 선택된 번호 제거
            available_numbers.pop(selected_idx)
            available_probs.pop(selected_idx)
            
            # 확률 재정규화
            if available_probs:
                total_prob = sum(available_probs)
                available_probs = [p / total_prob for p in available_probs]
        
        return sorted(selected_numbers)
    
    # === 유틸리티 함수들 ===
    
    def _is_prime(self, n: int) -> bool:
        """소수 판별"""
        if n < 2:
            return False
        if n == 2:
            return True
        if n % 2 == 0:
            return False
        for i in range(3, int(math.sqrt(n)) + 1, 2):
            if n % i == 0:
                return False
        return True
    
    def _is_fibonacci(self, n: int) -> bool:
        """피보나치 수 판별"""
        fib_numbers = [1, 1, 2, 3, 5, 8, 13, 21, 34]  # 45 이하 피보나치 수
        return n in fib_numbers
    
    def _is_frequent_number(self, number: int) -> bool:
        """자주 나오는 번호인지 판별"""
        frequency_data = self.stats.get('frequency_analysis', {})
        hot_numbers = frequency_data.get('hot_numbers', [])
        return number in hot_numbers
    
    def _get_recent_appearance(self, number: int) -> Optional[int]:
        """최근 출현 인덱스 반환"""
        for idx, row in self.data.iterrows():
            if number in row['winning_numbers']:
                return idx
        return None
    
    def _looks_random(self, number: int) -> bool:
        """무작위처럼 보이는지 판별"""
        # 간단한 휴리스틱: 특별한 의미가 없어 보이는 번호
        special_numbers = {1, 7, 8, 9, 10, 11, 13, 21, 22, 33, 42}
        return number not in special_numbers
    
    def _confirms_existing_beliefs(self, number: int) -> bool:
        """기존 믿음을 강화하는지 판별"""
        # 일반적으로 선호되거나 기피되는 번호
        preferred_numbers = {7, 8, 9, 11, 13}
        avoided_numbers = {4}
        return number in preferred_numbers or number in avoided_numbers
    
    def _is_memorable(self, number: int) -> bool:
        """기억하기 쉬운 번호인지 판별"""
        memorable_patterns = {
            1, 2, 3, 4, 5, 10, 11, 12, 20, 21, 22, 30, 33, 40, 44, 45
        }
        return number in memorable_patterns
    
    def _get_number_section(self, number: int) -> str:
        """번호 구간 반환"""
        if number <= 15:
            return 'low'
        elif number <= 30:
            return 'mid'
        else:
            return 'high'
    
    def _evaluate_system_balance(self, number: int) -> float:
        """시스템 전체 균형 평가"""
        # 전체 예측 시스템의 균형을 고려한 점수
        # 현재는 단순화된 버전
        return 5.0 + random.uniform(-0.5, 0.5)
    
    def _evaluate_prediction_diversity(self, number: int) -> float:
        """예측 다양성 평가"""
        # 예측의 다양성을 고려한 점수
        # 현재는 단순화된 버전
        return 5.0 + random.uniform(-0.5, 0.5)


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
            [1, 13, 21, 25, 28, 35]
        ],
        'bonus_number': [42, 35, 38, 42, 31],
        'number_sum': [150, 90, 134, 146, 123],
        'odd_count': [3, 3, 3, 5, 5],
        'even_count': [3, 3, 3, 1, 1],
        'low_count': [1, 3, 1, 0, 2],
        'mid_count': [2, 3, 4, 4, 3],
        'high_count': [3, 0, 1, 2, 1]
    })
    
    # 샘플 통계 데이터
    sample_stats = {
        'frequency_analysis': {
            'frequency_count': {i: random.randint(5, 15) for i in range(1, 46)},
            'hot_numbers': [13, 16, 23, 25, 28],
            'cold_numbers': [4, 11, 18, 33, 41]
        },
        'correlation_analysis': {
            'number_cooccurrence': {
                'top_companions': {
                    13: [(21, 5), (28, 4), (7, 3)],
                    21: [(13, 5), (35, 3), (1, 2)]
                }
            }
        },
        'distribution_analysis': {
            'section_distribution': {'low': 100, 'mid': 120, 'high': 80}
        }
    }
    
    print("=== 천재적 통찰 공식 테스트 (새로운 CSV 구조) ===")
    
    try:
        # 공식 초기화
        genius_formula = GeniusInsightFormula(sample_data, sample_stats)
        
        # GI 점수 계산
        results = genius_formula.calculate_genius_insight()
        
        # 결과 출력
        print(f"상위 5개 통찰 번호:")
        for number, score in results['top_insightful_numbers'][:5]:
            print(f"  번호 {number}: GI 점수 {score:.2f}")
        
        print(f"\n통찰 분석:")
        insights = results['insights_analysis']['pattern_insights']
        for insight in insights:
            print(f"  - {insight}")
        
        # 번호 예측 테스트
        predicted_numbers = genius_formula.predict_numbers(6)
        print(f"\n예측 번호: {predicted_numbers}")
        
        print("✅ 천재적 통찰 공식 테스트 완료")
        
    except Exception as e:
        print(f"❌ 테스트 중 오류: {e}")
        import traceback
        traceback.print_exc()
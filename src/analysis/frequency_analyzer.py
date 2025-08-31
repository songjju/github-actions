"""
파일명: src/analysis/frequency_analyzer.py
목적: 로또 번호의 다차원적 빈도 분석
작성자: AI Assistant
작성일: 2025-08-31
버전: 1.0

주요 클래스/함수:
- LottoFrequencyAnalyzer: 빈도 분석 메인 클래스
- NumberFrequencyAnalyzer: 개별 번호 빈도 분석
- CombinationFrequencyAnalyzer: 조합 빈도 분석
- PositionalFrequencyAnalyzer: 위치별 빈도 분석
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any, Optional, Set
from collections import Counter, defaultdict
import math
import statistics
import logging
from datetime import datetime, timedelta
from pathlib import Path
import sys
from itertools import combinations
from scipy import stats
import warnings

# 상위 디렉토리 imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.utils.config import Config, LottoConstants

warnings.filterwarnings('ignore')

class FrequencyAnalysisResult:
    """빈도 분석 결과를 담는 클래스"""
    
    def __init__(self, analysis_type: str):
        self.analysis_type = analysis_type
        self.frequency_data = {}
        self.statistics = {}
        self.insights = []
        self.predictions = {}
        self.confidence = 0.0
        
    def to_dict(self) -> Dict[str, Any]:
        return {
            'analysis_type': self.analysis_type,
            'frequency_data': self.frequency_data,
            'statistics': self.statistics,
            'insights': self.insights,
            'predictions': self.predictions,
            'confidence': self.confidence
        }


class LottoFrequencyAnalyzer:
    """로또 빈도 분석 메인 클래스"""
    
    def __init__(self, analysis_depth: str = 'comprehensive'):
        """
        초기화
        
        Args:
            analysis_depth (str): 분석 깊이 ('basic', 'standard', 'comprehensive')
        """
        self.analysis_depth = analysis_depth
        self.logger = self._setup_logger()
        
        # 서브 분석기들
        self.number_analyzer = NumberFrequencyAnalyzer()
        self.combination_analyzer = CombinationFrequencyAnalyzer()
        self.positional_analyzer = PositionalFrequencyAnalyzer()
        self.temporal_analyzer = TemporalFrequencyAnalyzer()
        
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
    
    def analyze_frequencies(self, df: pd.DataFrame) -> Dict[str, FrequencyAnalysisResult]:
        """
        종합적인 빈도 분석 수행
        
        Args:
            df (pd.DataFrame): 로또 데이터
            
        Returns:
            Dict[str, FrequencyAnalysisResult]: 분석 결과들
        """
        self.logger.info(f"빈도 분석 시작 (깊이: {self.analysis_depth})")
        
        results = {}
        
        try:
            # 1. 개별 번호 빈도 분석
            self.logger.info("개별 번호 빈도 분석")
            results['number_frequency'] = self.number_analyzer.analyze(df)
            
            # 2. 위치별 빈도 분석
            self.logger.info("위치별 빈도 분석")
            results['positional_frequency'] = self.positional_analyzer.analyze(df)
            
            if self.analysis_depth in ['standard', 'comprehensive']:
                # 3. 조합 빈도 분석
                self.logger.info("조합 빈도 분석")
                results['combination_frequency'] = self.combination_analyzer.analyze(df)
                
                # 4. 시간적 빈도 분석
                self.logger.info("시간적 빈도 분석")
                results['temporal_frequency'] = self.temporal_analyzer.analyze(df)
            
            if self.analysis_depth == 'comprehensive':
                # 5. 고급 패턴 빈도 분석
                self.logger.info("고급 패턴 빈도 분석")
                results['advanced_patterns'] = self._analyze_advanced_patterns(df)
                
                # 6. 교차 분석
                self.logger.info("교차 빈도 분석")
                results['cross_analysis'] = self._perform_cross_analysis(results)
            
            self.logger.info("빈도 분석 완료")
            return results
            
        except Exception as e:
            self.logger.error(f"빈도 분석 중 오류: {e}")
            raise
    
    def _analyze_advanced_patterns(self, df: pd.DataFrame) -> FrequencyAnalysisResult:
        """고급 패턴 빈도 분석"""
        result = FrequencyAnalysisResult('advanced_patterns')
        
        try:
            number_cols = [f'num{i}' for i in range(1, 7)]
            
            # 홀짝 패턴 빈도
            odd_even_patterns = Counter()
            # 구간 분포 패턴 빈도
            section_patterns = Counter()
            # 합계 범위 패턴 빈도
            sum_range_patterns = Counter()
            
            for idx, row in df.iterrows():
                numbers = [row[col] for col in number_cols if pd.notna(row[col])]
                if len(numbers) == 6:
                    # 홀짝 패턴
                    odd_count = sum(1 for n in numbers if n % 2 == 1)
                    odd_even_patterns[f"odd_{odd_count}"] += 1
                    
                    # 구간 분포 패턴
                    low = sum(1 for n in numbers if n <= 15)
                    mid = sum(1 for n in numbers if 16 <= n <= 30)
                    high = sum(1 for n in numbers if n >= 31)
                    section_patterns[f"low_{low}_mid_{mid}_high_{high}"] += 1
                    
                    # 합계 범위 패턴
                    total_sum = sum(numbers)
                    if total_sum <= 120:
                        sum_range = "low_sum"
                    elif total_sum <= 150:
                        sum_range = "medium_sum"
                    else:
                        sum_range = "high_sum"
                    sum_range_patterns[sum_range] += 1
            
            result.frequency_data = {
                'odd_even_patterns': dict(odd_even_patterns),
                'section_patterns': dict(section_patterns),
                'sum_range_patterns': dict(sum_range_patterns)
            }
            
            # 통계 계산
            result.statistics = {
                'most_common_odd_even': odd_even_patterns.most_common(1)[0] if odd_even_patterns else None,
                'most_common_section': section_patterns.most_common(1)[0] if section_patterns else None,
                'most_common_sum_range': sum_range_patterns.most_common(1)[0] if sum_range_patterns else None
            }
            
            # 인사이트 생성
            if odd_even_patterns:
                most_common_oe = odd_even_patterns.most_common(1)[0]
                oe_ratio = most_common_oe[1] / len(df)
                if oe_ratio > 0.3:
                    result.insights.append(f"홀짝 패턴 '{most_common_oe[0]}'이 {oe_ratio:.1%} 출현")
            
            result.confidence = 0.6  # 고급 패턴은 중간 신뢰도
            
        except Exception as e:
            result.insights.append(f"고급 패턴 분석 오류: {e}")
        
        return result
    
    def _perform_cross_analysis(self, results: Dict) -> FrequencyAnalysisResult:
        """교차 분석"""
        result = FrequencyAnalysisResult('cross_analysis')
        
        try:
            # 각 분석 결과 간의 상관관계 분석
            correlations = {}
            
            # 번호 빈도와 위치 빈도의 교차 분석
            if 'number_frequency' in results and 'positional_frequency' in results:
                num_freq = results['number_frequency'].frequency_data
                pos_freq = results['positional_frequency'].frequency_data
                
                # 번호별 선호 위치 분석
                number_position_preference = {}
                if 'individual_frequencies' in num_freq and 'position_preferences' in pos_freq:
                    for number in range(1, 46):
                        if number in num_freq['individual_frequencies']:
                            # 해당 번호가 어떤 위치에서 많이 나오는지
                            position_scores = {}
                            for pos in range(1, 7):
                                if f'position_{pos}' in pos_freq['position_preferences']:
                                    pos_data = pos_freq['position_preferences'][f'position_{pos}']
                                    if 'number_distribution' in pos_data:
                                        position_scores[pos] = pos_data['number_distribution'].get(number, 0)
                            
                            if position_scores:
                                preferred_pos = max(position_scores, key=position_scores.get)
                                number_position_preference[number] = {
                                    'preferred_position': preferred_pos,
                                    'score': position_scores[preferred_pos]
                                }
                
                correlations['number_position'] = number_position_preference
            
            result.frequency_data = correlations
            result.confidence = 0.4
            
        except Exception as e:
            result.insights.append(f"교차 분석 오류: {e}")
        
        return result


def generate_comprehensive_frequency_report(df: pd.DataFrame) -> Dict[str, Any]:
    """
    종합적인 빈도 분석 리포트 생성
    
    Args:
        df (pd.DataFrame): 로또 데이터
        
    Returns:
        Dict[str, Any]: 종합 리포트
    """
    analyzer = LottoFrequencyAnalyzer(analysis_depth='comprehensive')
    results = analyzer.analyze_frequencies(df)
    
    # 리포트 요약 생성
    report = {
        'executive_summary': _generate_executive_summary(results),
        'detailed_analysis': results,
        'recommendations': _generate_comprehensive_recommendations(results),
        'risk_assessment': _assess_prediction_risks(results),
        'data_quality': _assess_data_quality(df, results)
    }
    
    return report


def _generate_executive_summary(results: Dict) -> Dict[str, Any]:
    """경영진 요약 생성"""
    summary = {
        'key_findings': [],
        'confidence_scores': {},
        'top_insights': [],
        'prediction_strength': 'medium'
    }
    
    # 각 분석의 신뢰도 수집
    for analysis_type, result in results.items():
        if hasattr(result, 'confidence'):
            summary['confidence_scores'][analysis_type] = result.confidence
    
    # 평균 신뢰도 계산
    if summary['confidence_scores']:
        avg_confidence = np.mean(list(summary['confidence_scores'].values()))
        if avg_confidence > 0.7:
            summary['prediction_strength'] = 'high'
        elif avg_confidence < 0.4:
            summary['prediction_strength'] = 'low'
    
    # 주요 발견사항 수집
    all_insights = []
    for result in results.values():
        if hasattr(result, 'insights'):
            all_insights.extend(result.insights)
    
    # 상위 5개 인사이트
    summary['top_insights'] = all_insights[:5]
    
    # 핵심 발견사항
    if 'number_frequency' in results:
        num_result = results['number_frequency']
        if hasattr(num_result, 'frequency_data') and 'hot_numbers' in num_result.frequency_data:
            hot_count = len(num_result.frequency_data['hot_numbers'])
            summary['key_findings'].append(f"핫 번호 {hot_count}개 식별")
    
    return summary


def _generate_comprehensive_recommendations(results: Dict) -> List[str]:
    """종합 권장사항 생성"""
    recommendations = []
    
    # 번호 선택 권장사항
    if 'number_frequency' in results:
        num_result = results['number_frequency']
        if hasattr(num_result, 'frequency_data'):
            freq_data = num_result.frequency_data
            
            if 'hot_numbers' in freq_data and len(freq_data['hot_numbers']) > 0:
                recommendations.append(f"핫 번호 활용: {freq_data['hot_numbers'][:3]} 등을 고려하세요.")
            
            if 'recent_analysis' in freq_data:
                recent = freq_data['recent_analysis']
                if 'trending_up' in recent and len(recent['trending_up']) > 0:
                    recommendations.append(f"상승 트렌드 번호 주목: {recent['trending_up'][:3]}")
    
    # 위치별 권장사항
    if 'positional_frequency' in results:
        pos_result = results['positional_frequency']
        if hasattr(pos_result, 'insights') and len(pos_result.insights) > 0:
            recommendations.append(f"위치별 패턴 활용: {pos_result.insights[0]}")
    
    # 시간적 권장사항
    if 'temporal_frequency' in results:
        temp_result = results['temporal_frequency']
        if hasattr(temp_result, 'insights') and len(temp_result.insights) > 0:
            recommendations.append(f"시간적 패턴 고려: {temp_result.insights[0]}")
    
    # 조합 권장사항
    if 'combination_frequency' in results:
        comb_result = results['combination_frequency']
        if hasattr(comb_result, 'frequency_data') and 'top_pairs' in comb_result.frequency_data:
            top_pairs = comb_result.frequency_data['top_pairs']
            if top_pairs and len(top_pairs) > 0:
                best_pair = top_pairs[0][0]  # (pair, frequency)
                recommendations.append(f"효과적인 번호 조합: {best_pair} 등의 조합 고려")
    
    # 일반적인 권장사항
    if not recommendations:
        recommendations.extend([
            "데이터 기반 분석을 통해 번호를 선택하세요.",
            "과거 패턴을 참고하되, 확률의 독립성을 고려하세요.",
            "다양한 분석 기법을 종합적으로 활용하세요."
        ])
    
    return recommendations


def _assess_prediction_risks(results: Dict) -> Dict[str, Any]:
    """예측 위험 평가"""
    risk_assessment = {
        'overall_risk': 'medium',
        'confidence_variance': 0,
        'data_consistency': 'good',
        'risk_factors': [],
        'mitigation_strategies': []
    }
    
    # 신뢰도 분산 계산
    confidences = []
    for result in results.values():
        if hasattr(result, 'confidence'):
            confidences.append(result.confidence)
    
    if len(confidences) > 1:
        risk_assessment['confidence_variance'] = np.var(confidences)
        
        if np.var(confidences) > 0.1:
            risk_assessment['risk_factors'].append("분석 간 신뢰도 편차가 큼")
            risk_assessment['mitigation_strategies'].append("여러 분석 결과를 균형있게 활용")
    
    # 전체 위험도 평가
    if confidences:
        avg_confidence = np.mean(confidences)
        if avg_confidence < 0.3:
            risk_assessment['overall_risk'] = 'high'
            risk_assessment['risk_factors'].append("낮은 예측 신뢰도")
        elif avg_confidence > 0.7:
            risk_assessment['overall_risk'] = 'low'
    
    # 데이터 품질 기반 위험 요소
    if len(results) < 3:
        risk_assessment['risk_factors'].append("제한된 분석 범위")
        risk_assessment['mitigation_strategies'].append("더 다양한 분석 기법 적용 필요")
    
    return risk_assessment


def _assess_data_quality(df: pd.DataFrame, results: Dict) -> Dict[str, Any]:
    """데이터 품질 평가"""
    quality_assessment = {
        'data_completeness': 0,
        'data_consistency': 'good',
        'temporal_coverage': 'adequate',
        'quality_score': 0,
        'quality_issues': []
    }
    
    # 데이터 완성도
    total_cells = df.shape[0] * df.shape[1]
    non_null_cells = df.count().sum()
    completeness = non_null_cells / total_cells if total_cells > 0 else 0
    quality_assessment['data_completeness'] = completeness
    
    # 시간적 범위
    if 'draw_date' in df.columns:
        try:
            df['draw_date'] = pd.to_datetime(df['draw_date'])
            date_range = (df['draw_date'].max() - df['draw_date'].min()).days
            
            if date_range < 365:
                quality_assessment['temporal_coverage'] = 'limited'
                quality_assessment['quality_issues'].append("시간적 범위가 제한적 (1년 미만)")
            elif date_range > 365 * 3:
                quality_assessment['temporal_coverage'] = 'excellent'
        except:
            quality_assessment['quality_issues'].append("날짜 데이터 처리 오류")
    
    # 품질 점수 계산
    score = completeness * 0.4
    
    if quality_assessment['temporal_coverage'] == 'excellent':
        score += 0.3
    elif quality_assessment['temporal_coverage'] == 'adequate':
        score += 0.2
    
    if len(quality_assessment['quality_issues']) == 0:
        score += 0.3
    elif len(quality_assessment['quality_issues']) <= 2:
        score += 0.15
    
    quality_assessment['quality_score'] = min(1.0, score)
    
    return quality_assessment

class NumberFrequencyAnalyzer:
    """개별 번호 빈도 분석기"""
    
    def analyze(self, df: pd.DataFrame) -> FrequencyAnalysisResult:
        """개별 번호 빈도 분석"""
        result = FrequencyAnalysisResult('number_frequency')
        
        try:
            number_cols = [f'num{i}' for i in range(1, 7)]
            if 'bonus' in df.columns:
                number_cols.append('bonus')
            
            # 개별 번호 출현 빈도
            individual_frequencies = Counter()
            
            for col in number_cols:
                if col in df.columns:
                    for number in df[col].dropna():
                        individual_frequencies[int(number)] += 1
            
            # 통계 계산
            frequencies = list(individual_frequencies.values())
            if frequencies:
                freq_mean = np.mean(frequencies)
                freq_std = np.std(frequencies)
                freq_cv = freq_std / freq_mean if freq_mean > 0 else 0
                
                # 핫/콜드 번호 분류
                hot_threshold = freq_mean + freq_std
                cold_threshold = freq_mean - freq_std
                
                hot_numbers = [num for num, freq in individual_frequencies.items() 
                              if freq > hot_threshold]
                cold_numbers = [num for num, freq in individual_frequencies.items() 
                               if freq < cold_threshold]
                
                # 최근 출현 분석
                recent_analysis = self._analyze_recent_appearances(df, individual_frequencies)
                
                result.frequency_data = {
                    'individual_frequencies': dict(individual_frequencies),
                    'hot_numbers': hot_numbers,
                    'cold_numbers': cold_numbers,
                    'recent_analysis': recent_analysis
                }
                
                result.statistics = {
                    'total_draws': len(df),
                    'mean_frequency': freq_mean,
                    'std_frequency': freq_std,
                    'coefficient_of_variation': freq_cv,
                    'most_frequent': individual_frequencies.most_common(5),
                    'least_frequent': individual_frequencies.most_common()[-5:],
                    'hot_numbers_count': len(hot_numbers),
                    'cold_numbers_count': len(cold_numbers)
                }
                
                # 인사이트 생성
                self._generate_frequency_insights(result)
                
                # 예측값 계산
                result.predictions = self._calculate_frequency_predictions(individual_frequencies, recent_analysis)
                result.confidence = min(0.8, freq_cv) if freq_cv > 0 else 0.5
        
        except Exception as e:
            result.insights.append(f"번호 빈도 분석 오류: {e}")
        
        return result
    
    def _analyze_recent_appearances(self, df: pd.DataFrame, frequencies: Counter) -> Dict:
        """최근 출현 분석"""
        number_cols = [f'num{i}' for i in range(1, 7)]
        
        # 최근 20회 분석
        recent_20 = df.tail(20) if len(df) >= 20 else df
        recent_frequencies = Counter()
        
        for col in number_cols:
            if col in recent_20.columns:
                for number in recent_20[col].dropna():
                    recent_frequencies[int(number)] += 1
        
        # 트렌드 분석
        trending_up = []
        trending_down = []
        
        for number in range(1, 46):
            total_freq = frequencies.get(number, 0)
            recent_freq = recent_frequencies.get(number, 0)
            
            # 전체 평균 대비 최근 빈도
            expected_recent = total_freq * (len(recent_20) / len(df)) if len(df) > 0 else 0
            
            if recent_freq > expected_recent * 1.5:  # 50% 이상 증가
                trending_up.append(number)
            elif recent_freq < expected_recent * 0.5:  # 50% 이상 감소
                trending_down.append(number)
        
        return {
            'recent_frequencies': dict(recent_frequencies),
            'trending_up': trending_up,
            'trending_down': trending_down,
            'recent_period': len(recent_20)
        }
    
    def _generate_frequency_insights(self, result: FrequencyAnalysisResult):
        """빈도 기반 인사이트 생성"""
        stats = result.statistics
        freq_data = result.frequency_data
        
        # 1. 편차 분석
        cv = stats.get('coefficient_of_variation', 0)
        if cv > 0.3:
            result.insights.append(f"번호 출현 빈도의 편차가 큼 (CV: {cv:.3f})")
        elif cv < 0.1:
            result.insights.append(f"번호 출현 빈도가 균등함 (CV: {cv:.3f})")
        
        # 2. 핫/콜드 번호 분석
        hot_count = stats.get('hot_numbers_count', 0)
        cold_count = stats.get('cold_numbers_count', 0)
        
        if hot_count > 10:
            result.insights.append(f"자주 출현하는 번호가 많음 ({hot_count}개)")
        if cold_count > 10:
            result.insights.append(f"드물게 출현하는 번호가 많음 ({cold_count}개)")
        
        # 3. 최근 트렌드 분석
        recent = freq_data.get('recent_analysis', {})
        trending_up = recent.get('trending_up', [])
        trending_down = recent.get('trending_down', [])
        
        if trending_up:
            result.insights.append(f"최근 상승 트렌드 번호: {trending_up[:5]}")
        if trending_down:
            result.insights.append(f"최근 하락 트렌드 번호: {trending_down[:5]}")
    
    def _calculate_frequency_predictions(self, frequencies: Counter, recent_analysis: Dict) -> Dict:
        """빈도 기반 예측 계산"""
        predictions = {}
        
        # 1. 기본 빈도 기반 예측
        total_count = sum(frequencies.values())
        if total_count > 0:
            frequency_scores = {}
            for number in range(1, 46):
                freq = frequencies.get(number, 0)
                frequency_scores[number] = freq / total_count
            
            predictions['frequency_based'] = frequency_scores
        
        # 2. 트렌드 조정 예측
        trending_up = recent_analysis.get('trending_up', [])
        trending_down = recent_analysis.get('trending_down', [])
        
        trend_scores = {}
        for number in range(1, 46):
            base_score = predictions.get('frequency_based', {}).get(number, 0)
            
            if number in trending_up:
                trend_scores[number] = base_score * 1.2  # 20% 보너스
            elif number in trending_down:
                trend_scores[number] = base_score * 0.8  # 20% 패널티
            else:
                trend_scores[number] = base_score
        
        predictions['trend_adjusted'] = trend_scores
        
        return predictions


class PositionalFrequencyAnalyzer:
    """위치별 빈도 분석기"""
    
    def analyze(self, df: pd.DataFrame) -> FrequencyAnalysisResult:
        """위치별 빈도 분석"""
        result = FrequencyAnalysisResult('positional_frequency')
        
        try:
            number_cols = [f'num{i}' for i in range(1, 7)]
            if not all(col in df.columns for col in number_cols):
                return result
            
            position_preferences = {}
            
            for i, col in enumerate(number_cols, 1):
                position_data = {}
                numbers = df[col].dropna()
                
                if len(numbers) > 0:
                    # 위치별 번호 분포
                    number_distribution = Counter(numbers.astype(int))
                    
                    # 위치별 통계
                    position_stats = {
                        'mean': numbers.mean(),
                        'median': numbers.median(),
                        'std': numbers.std(),
                        'min': numbers.min(),
                        'max': numbers.max(),
                        'range': numbers.max() - numbers.min()
                    }
                    
                    # 구간별 선호도
                    section_preferences = {
                        'low_section': ((numbers <= 15).sum() / len(numbers)),
                        'mid_section': (((numbers >= 16) & (numbers <= 30)).sum() / len(numbers)),
                        'high_section': ((numbers >= 31).sum() / len(numbers))
                    }
                    
                    # 홀짝 선호도
                    odd_preference = (numbers % 2 == 1).sum() / len(numbers)
                    
                    position_data = {
                        'number_distribution': dict(number_distribution),
                        'statistics': position_stats,
                        'section_preferences': section_preferences,
                        'odd_preference': odd_preference,
                        'sample_size': len(numbers)
                    }
                    
                    position_preferences[f'position_{i}'] = position_data
            
            result.frequency_data = {
                'position_preferences': position_preferences
            }
            
            # 위치별 차이 분석
            position_stats = self._analyze_position_differences(position_preferences)
            result.statistics = position_stats
            
            # 인사이트 생성
            self._generate_positional_insights(result, position_preferences)
            
            result.confidence = 0.7
        
        except Exception as e:
            result.insights.append(f"위치별 분석 오류: {e}")
        
        return result
    
    def _analyze_position_differences(self, position_prefs: Dict) -> Dict:
        """위치별 차이 분석"""
        means = []
        stds = []
        ranges = []
        
        for pos_data in position_prefs.values():
            stats_data = pos_data.get('statistics', {})
            means.append(stats_data.get('mean', 0))
            stds.append(stats_data.get('std', 0))
            ranges.append(stats_data.get('range', 0))
        
        return {
            'position_mean_differences': {
                'values': means,
                'std_of_means': np.std(means) if means else 0,
                'range_of_means': max(means) - min(means) if means else 0
            },
            'position_variability': {
                'avg_std': np.mean(stds) if stds else 0,
                'std_of_stds': np.std(stds) if stds else 0
            },
            'position_ranges': {
                'avg_range': np.mean(ranges) if ranges else 0,
                'max_range': max(ranges) if ranges else 0
            }
        }
    
    def _generate_positional_insights(self, result: FrequencyAnalysisResult, position_prefs: Dict):
        """위치별 인사이트 생성"""
        for pos_name, pos_data in position_prefs.items():
            stats = pos_data.get('statistics', {})
            sections = pos_data.get('section_preferences', {})
            
            # 위치별 선호 구간
            max_section = max(sections, key=sections.get) if sections else None
            max_ratio = sections.get(max_section, 0) if max_section else 0
            
            if max_ratio > 0.5:
                result.insights.append(f"{pos_name}: {max_section} 선호 ({max_ratio:.1%})")
            
            # 위치별 홀짝 선호
            odd_pref = pos_data.get('odd_preference', 0.5)
            if odd_pref > 0.7:
                result.insights.append(f"{pos_name}: 홀수 선호 ({odd_pref:.1%})")
            elif odd_pref < 0.3:
                result.insights.append(f"{pos_name}: 짝수 선호 ({odd_pref:.1%})")


class CombinationFrequencyAnalyzer:
    """조합 빈도 분석기"""
    
    def analyze(self, df: pd.DataFrame) -> FrequencyAnalysisResult:
        """조합 빈도 분석"""
        result = FrequencyAnalysisResult('combination_frequency')
        
        try:
            number_cols = [f'num{i}' for i in range(1, 7)]
            if not all(col in df.columns for col in number_cols):
                return result
            
            # 2개 조합 빈도
            pair_frequencies = Counter()
            # 3개 조합 빈도 (상위 조합만)
            triple_frequencies = Counter()
            
            for idx, row in df.iterrows():
                numbers = [row[col] for col in number_cols if pd.notna(row[col])]
                if len(numbers) >= 2:
                    # 2개 조합
                    for pair in combinations(numbers, 2):
                        sorted_pair = tuple(sorted(pair))
                        pair_frequencies[sorted_pair] += 1
                    
                    # 3개 조합 (너무 많아서 상위만)
                    if len(numbers) >= 3:
                        for triple in combinations(numbers, 3):
                            sorted_triple = tuple(sorted(triple))
                            triple_frequencies[sorted_triple] += 1
            
            # 상위 조합들만 저장 (메모리 효율성)
            top_pairs = pair_frequencies.most_common(50)
            top_triples = triple_frequencies.most_common(20)
            
            result.frequency_data = {
                'top_pairs': top_pairs,
                'top_triples': top_triples,
                'total_unique_pairs': len(pair_frequencies),
                'total_unique_triples': len(triple_frequencies)
            }
            
            # 통계 계산
            if pair_frequencies:
                pair_freq_values = list(pair_frequencies.values())
                result.statistics = {
                    'pair_frequency_mean': np.mean(pair_freq_values),
                    'pair_frequency_std': np.std(pair_freq_values),
                    'max_pair_frequency': max(pair_freq_values),
                    'pairs_appearing_multiple_times': sum(1 for freq in pair_freq_values if freq > 1)
                }
            
            # 인사이트 생성
            self._generate_combination_insights(result)
            
            result.confidence = 0.5
        
        except Exception as e:
            result.insights.append(f"조합 분석 오류: {e}")
        
        return result
    
    def _generate_combination_insights(self, result: FrequencyAnalysisResult):
        """조합 분석 인사이트 생성"""
        freq_data = result.frequency_data
        stats = result.statistics
        
        # 반복 출현 조합 분석
        pairs_multi = stats.get('pairs_appearing_multiple_times', 0)
        total_pairs = freq_data.get('total_unique_pairs', 0)
        
        if total_pairs > 0:
            repeat_ratio = pairs_multi / total_pairs
            if repeat_ratio > 0.1:  # 10% 이상이 반복 출현
                result.insights.append(f"반복 출현 조합 비율: {repeat_ratio:.1%}")
        
        # 최고 빈도 조합 분석
        top_pairs = freq_data.get('top_pairs', [])
        if top_pairs:
            best_pair, best_freq = top_pairs[0]
            if best_freq >= 3:
                result.insights.append(f"최고 빈도 조합: {best_pair} ({best_freq}회)")


class TemporalFrequencyAnalyzer:
    """시간적 빈도 분석기"""
    
    def analyze(self, df: pd.DataFrame) -> FrequencyAnalysisResult:
        """시간적 빈도 분석"""
        result = FrequencyAnalysisResult('temporal_frequency')
        
        try:
            if 'draw_date' not in df.columns:
                result.insights.append("날짜 정보가 없어 시간적 분석 불가")
                return result
            
            df['draw_date'] = pd.to_datetime(df['draw_date'])
            number_cols = [f'num{i}' for i in range(1, 7)]
            
            # 월별 빈도 분석
            monthly_analysis = self._analyze_monthly_frequencies(df, number_cols)
            
            # 계절별 빈도 분석
            seasonal_analysis = self._analyze_seasonal_frequencies(df, number_cols)
            
            # 연도별 빈도 분석
            yearly_analysis = self._analyze_yearly_frequencies(df, number_cols)
            
            # 주기성 분석
            periodicity_analysis = self._analyze_periodicity(df, number_cols)
            
            result.frequency_data = {
                'monthly_analysis': monthly_analysis,
                'seasonal_analysis': seasonal_analysis,
                'yearly_analysis': yearly_analysis,
                'periodicity_analysis': periodicity_analysis
            }
            
            # 시간적 트렌드 통계
            result.statistics = self._calculate_temporal_statistics(
                monthly_analysis, seasonal_analysis, yearly_analysis
            )
            
            # 인사이트 생성
            self._generate_temporal_insights(result)
            
            result.confidence = 0.6
        
        except Exception as e:
            result.insights.append(f"시간적 분석 오류: {e}")
        
        return result
    
    def _analyze_monthly_frequencies(self, df: pd.DataFrame, number_cols: List[str]) -> Dict:
        """월별 빈도 분석"""
        df['month'] = df['draw_date'].dt.month
        monthly_frequencies = {}
        
        for month in range(1, 13):
            month_data = df[df['month'] == month]
            month_numbers = []
            
            for col in number_cols:
                if col in month_data.columns:
                    month_numbers.extend(month_data[col].dropna().astype(int).tolist())
            
            if month_numbers:
                monthly_frequencies[month] = {
                    'total_numbers': len(month_numbers),
                    'unique_numbers': len(set(month_numbers)),
                    'average_number': np.mean(month_numbers),
                    'most_common': Counter(month_numbers).most_common(5),
                    'draws_count': len(month_data)
                }
        
        return monthly_frequencies
    
    def _analyze_seasonal_frequencies(self, df: pd.DataFrame, number_cols: List[str]) -> Dict:
        """계절별 빈도 분석"""
        # 계절 정의 (한국 기준)
        season_map = {
            12: 'winter', 1: 'winter', 2: 'winter',
            3: 'spring', 4: 'spring', 5: 'spring',
            6: 'summer', 7: 'summer', 8: 'summer',
            9: 'autumn', 10: 'autumn', 11: 'autumn'
        }
        
        df['season'] = df['draw_date'].dt.month.map(season_map)
        seasonal_frequencies = {}
        
        for season in ['spring', 'summer', 'autumn', 'winter']:
            season_data = df[df['season'] == season]
            season_numbers = []
            
            for col in number_cols:
                if col in season_data.columns:
                    season_numbers.extend(season_data[col].dropna().astype(int).tolist())
            
            if season_numbers:
                seasonal_frequencies[season] = {
                    'total_numbers': len(season_numbers),
                    'average_number': np.mean(season_numbers),
                    'number_distribution': Counter(season_numbers),
                    'draws_count': len(season_data)
                }
        
        return seasonal_frequencies
    
    def _analyze_yearly_frequencies(self, df: pd.DataFrame, number_cols: List[str]) -> Dict:
        """연도별 빈도 분석"""
        df['year'] = df['draw_date'].dt.year
        yearly_frequencies = {}
        
        for year in df['year'].unique():
            if pd.notna(year):
                year_data = df[df['year'] == year]
                year_numbers = []
                
                for col in number_cols:
                    if col in year_data.columns:
                        year_numbers.extend(year_data[col].dropna().astype(int).tolist())
                
                if year_numbers:
                    yearly_frequencies[int(year)] = {
                        'total_numbers': len(year_numbers),
                        'average_number': np.mean(year_numbers),
                        'most_frequent': Counter(year_numbers).most_common(10),
                        'draws_count': len(year_data)
                    }
        
        return yearly_frequencies
    
    def _analyze_periodicity(self, df: pd.DataFrame, number_cols: List[str]) -> Dict:
        """주기성 분석"""
        # 각 번호의 출현 간격 분석
        periodicity_data = {}
        
        for number in range(1, 46):
            appearance_indices = []
            
            for idx, row in df.iterrows():
                if any(row[col] == number for col in number_cols if pd.notna(row[col])):
                    appearance_indices.append(idx)
            
            if len(appearance_indices) >= 3:
                gaps = [appearance_indices[i+1] - appearance_indices[i] 
                       for i in range(len(appearance_indices)-1)]
                
                if gaps:
                    periodicity_data[number] = {
                        'appearances': len(appearance_indices),
                        'average_gap': np.mean(gaps),
                        'gap_std': np.std(gaps),
                        'gap_consistency': 1 - (np.std(gaps) / np.mean(gaps)) if np.mean(gaps) > 0 else 0,
                        'gaps': gaps
                    }
        
        return periodicity_data
    
    def _calculate_temporal_statistics(self, monthly: Dict, seasonal: Dict, yearly: Dict) -> Dict:
        """시간적 통계 계산"""
        stats = {}
        
        # 월별 변동성
        if monthly:
            monthly_avgs = [data['average_number'] for data in monthly.values()]
            stats['monthly_variability'] = np.std(monthly_avgs) if monthly_avgs else 0
        
        # 계절별 변동성
        if seasonal:
            seasonal_avgs = [data['average_number'] for data in seasonal.values()]
            stats['seasonal_variability'] = np.std(seasonal_avgs) if seasonal_avgs else 0
        
        # 연도별 트렌드
        if yearly and len(yearly) > 1:
            years = sorted(yearly.keys())
            yearly_avgs = [yearly[year]['average_number'] for year in years]
            
            # 선형 트렌드
            x = np.arange(len(yearly_avgs))
            if len(yearly_avgs) > 1:
                slope, intercept, r_value, p_value, std_err = stats.linregress(x, yearly_avgs)
                stats['yearly_trend'] = {
                    'slope': slope,
                    'correlation': r_value,
                    'p_value': p_value,
                    'trend_direction': 'increasing' if slope > 0 else 'decreasing'
                }
        
        return stats
    
    def _generate_temporal_insights(self, result: FrequencyAnalysisResult):
        """시간적 인사이트 생성"""
        stats = result.statistics
        freq_data = result.frequency_data
        
        # 월별 변동성
        monthly_var = stats.get('monthly_variability', 0)
        if monthly_var > 3:
            result.insights.append(f"월별 번호 평균의 변동이 큼 (표준편차: {monthly_var:.1f})")
        
        # 계절별 변동성
        seasonal_var = stats.get('seasonal_variability', 0)
        if seasonal_var > 2:
            result.insights.append(f"계절별 번호 평균의 변동이 있음 (표준편차: {seasonal_var:.1f})")
        
        # 연도별 트렌드
        yearly_trend = stats.get('yearly_trend', {})
        if yearly_trend:
            slope = yearly_trend.get('slope', 0)
            p_value = yearly_trend.get('p_value', 1)
            
            if p_value < 0.05 and abs(slope) > 0.1:
                direction = yearly_trend.get('trend_direction', 'stable')
                result.insights.append(f"연도별 번호 평균이 {direction} 추세 (기울기: {slope:.3f})")


# === 모듈 테스트 ===
if __name__ == "__main__":
    # 테스트용 데이터 생성
    np.random.seed(42)
    
    test_data = []
    base_date = datetime(2020, 1, 1)
    
    for i in range(100):
        # 일부 패턴을 의도적으로 삽입
        if i % 20 == 0:  # 특정 번호 조합을 더 자주 사용
            numbers = [1, 7, 14, 21, 28, 35]
        elif i % 15 == 0:  # 홀수 선호 패턴
            numbers = sorted(np.random.choice([1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23], size=6, replace=False))
        else:
            numbers = sorted(np.random.choice(range(1, 46), size=6, replace=False))
        
        row = {
            'round': 1000 + i,
            'draw_date': base_date + timedelta(days=i*3),
        }
        
        for j, num in enumerate(numbers, 1):
            row[f'num{j}'] = num
        
        test_data.append(row)
    
    test_df = pd.DataFrame(test_data)
    
    print("=== 빈도 분석기 테스트 ===")
    print(f"테스트 데이터: {len(test_df)}행")
    
    try:
        # 빈도 분석기 초기화 및 실행
        analyzer = LottoFrequencyAnalyzer(analysis_depth='comprehensive')
        results = analyzer.analyze_frequencies(test_df)
        
        print(f"\n📊 빈도 분석 결과:")
        
        for analysis_type, result in results.items():
            print(f"\n{analysis_type.upper()}:")
            print(f"  신뢰도: {result.confidence:.3f}")
            
            if result.statistics:
                print("  주요 통계:")
                for key, value in list(result.statistics.items())[:3]:
                    if isinstance(value, (int, float)):
                        print(f"    {key}: {value:.3f}" if isinstance(value, float) else f"    {key}: {value}")
                    else:
                        print(f"    {key}: {str(value)[:50]}...")
            
            if result.insights:
                print("  주요 인사이트:")
                for insight in result.insights[:3]:
                    print(f"    - {insight}")
        
        # 종합 리포트 테스트
        print(f"\n📋 종합 리포트 테스트:")
        comprehensive_report = generate_comprehensive_frequency_report(test_df)
        
        print("  경영진 요약:")
        exec_summary = comprehensive_report['executive_summary']
        print(f"    예측 강도: {exec_summary['prediction_strength']}")
        print(f"    핵심 발견: {len(exec_summary['key_findings'])}개")
        
        print("  권장사항:")
        recommendations = comprehensive_report['recommendations']
        for i, rec in enumerate(recommendations[:3], 1):
            print(f"    {i}. {rec}")
        
        print("  위험 평가:")
        risk_assessment = comprehensive_report['risk_assessment']
        print(f"    전체 위험도: {risk_assessment['overall_risk']}")
        print(f"    위험 요소: {len(risk_assessment['risk_factors'])}개")
        
        print("  데이터 품질:")
        data_quality = comprehensive_report['data_quality']
        print(f"    완성도: {data_quality['data_completeness']:.1%}")
        print(f"    품질 점수: {data_quality['quality_score']:.3f}")
        
        # 개별 분석기 테스트
        print(f"\n🔍 개별 분석기 테스트:")
        
        # 번호 빈도 분석
        num_result = analyzer.number_analyzer.analyze(test_df)
        print(f"  번호 빈도 분석: 신뢰도 {num_result.confidence:.3f}")
        if 'hot_numbers' in num_result.frequency_data:
            hot_nums = num_result.frequency_data['hot_numbers']
            print(f"    핫 번호: {hot_nums[:5]}")
        
        # 위치별 분석
        pos_result = analyzer.positional_analyzer.analyze(test_df)
        print(f"  위치별 분석: 신뢰도 {pos_result.confidence:.3f}")
        print(f"    인사이트 수: {len(pos_result.insights)}")
        
        # 조합 분석
        comb_result = analyzer.combination_analyzer.analyze(test_df)
        print(f"  조합 분석: 신뢰도 {comb_result.confidence:.3f}")
        if 'top_pairs' in comb_result.frequency_data:
            top_pairs = comb_result.frequency_data['top_pairs']
            if top_pairs:
                print(f"    최고 빈도 조합: {top_pairs[0]}")
        
        # 시간적 분석
        temp_result = analyzer.temporal_analyzer.analyze(test_df)
        print(f"  시간적 분석: 신뢰도 {temp_result.confidence:.3f}")
        print(f"    인사이트 수: {len(temp_result.insights)}")
        
        print("\n✅ 빈도 분석기 테스트 완료")
        
    except Exception as e:
        print(f"❌ 테스트 중 오류: {e}")
        import traceback
        traceback.print_exc()
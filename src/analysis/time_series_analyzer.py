"""
파일명: src/analysis/time_series_analyzer.py
목적: 로또 데이터의 시계열 분석 및 예측
작성자: AI Assistant
작성일: 2025-08-31
버전: 1.0

주요 클래스/함수:
- LottoTimeSeriesAnalyzer: 시계열 분석 메인 클래스
- TrendAnalyzer: 트렌드 분석
- SeasonalityDetector: 계절성 탐지
- CyclicalPatternAnalyzer: 순환 패턴 분석
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
from scipy import stats
from scipy.fft import fft, fftfreq
from scipy.signal import find_peaks, savgol_filter
import warnings

# 상위 디렉토리 imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.utils.config import Config, LottoConstants

warnings.filterwarnings('ignore')

class TimeSeriesResult:
    """시계열 분석 결과를 담는 클래스"""
    
    def __init__(self, analysis_type: str):
        self.analysis_type = analysis_type
        self.time_series_data = {}
        self.trends = {}
        self.seasonal_components = {}
        self.forecasts = {}
        self.confidence_intervals = {}
        self.model_performance = {}
        self.insights = []
        
    def to_dict(self) -> Dict[str, Any]:
        return {
            'analysis_type': self.analysis_type,
            'time_series_data': self.time_series_data,
            'trends': self.trends,
            'seasonal_components': self.seasonal_components,
            'forecasts': self.forecasts,
            'confidence_intervals': self.confidence_intervals,
            'model_performance': self.model_performance,
            'insights': self.insights
        }


class LottoTimeSeriesAnalyzer:
    """로또 시계열 분석 메인 클래스"""
    
    def __init__(self, forecast_horizon: int = 10):
        """
        초기화
        
        Args:
            forecast_horizon (int): 예측 기간 (회차 수)
        """
        self.forecast_horizon = forecast_horizon
        self.logger = self._setup_logger()
        
        # 서브 분석기들
        self.trend_analyzer = TrendAnalyzer()
        self.seasonality_detector = SeasonalityDetector()
        self.cyclical_analyzer = CyclicalPatternAnalyzer()
        self.forecast_engine = ForecastEngine()
        
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
    
    def analyze_time_series(self, df: pd.DataFrame) -> Dict[str, TimeSeriesResult]:
        """
        종합적인 시계열 분석
        
        Args:
            df (pd.DataFrame): 로또 데이터 (시간 순서 정렬 필요)
            
        Returns:
            Dict[str, TimeSeriesResult]: 분석 결과들
        """
        self.logger.info("시계열 분석 시작")
        
        results = {}
        
        try:
            # 데이터 전처리
            df_sorted = self._prepare_time_series_data(df)
            
            # 1. 개별 번호 시계열 분석
            self.logger.info("개별 번호 시계열 분석")
            results['number_trends'] = self._analyze_number_time_series(df_sorted)
            
            # 2. 집계 지표 시계열 분석
            self.logger.info("집계 지표 시계열 분석")
            results['aggregate_trends'] = self._analyze_aggregate_time_series(df_sorted)
            
            # 3. 주기성 분석
            self.logger.info("주기성 분석")
            results['periodicity'] = self.seasonality_detector.detect_seasonality(df_sorted)
            
            # 4. 순환 패턴 분석
            self.logger.info("순환 패턴 분석")
            results['cyclical_patterns'] = self.cyclical_analyzer.analyze_cycles(df_sorted)
            
            # 5. 예측 모델링
            self.logger.info("예측 모델링")
            results['forecasts'] = self.forecast_engine.generate_forecasts(
                df_sorted, self.forecast_horizon
            )
            
            self.logger.info("시계열 분석 완료")
            return results
            
        except Exception as e:
            self.logger.error(f"시계열 분석 중 오류: {e}")
            raise
    
    def _prepare_time_series_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """시계열 분석용 데이터 준비"""
        df_prepared = df.copy()
        
        # 날짜 컬럼 처리
        if 'draw_date' in df.columns:
            df_prepared['draw_date'] = pd.to_datetime(df_prepared['draw_date'])
            df_prepared = df_prepared.sort_values('draw_date')
        elif 'round' in df.columns:
            df_prepared = df_prepared.sort_values('round')
        
        # 시간 인덱스 생성
        df_prepared['time_index'] = range(len(df_prepared))
        
        return df_prepared
    
    def _analyze_number_time_series(self, df: pd.DataFrame) -> TimeSeriesResult:
        """개별 번호의 시계열 분석"""
        result = TimeSeriesResult('number_trends')
        
        try:
            number_cols = [f'num{i}' for i in range(1, 7)]
            
            # 각 번호의 출현 시계열 생성
            number_time_series = {}
            
            for number in range(1, 46):
                # 해당 번호의 출현 여부를 시계열로 변환
                appearances = []
                for idx, row in df.iterrows():
                    is_present = any(row[col] == number for col in number_cols if pd.notna(row[col]))
                    appearances.append(1 if is_present else 0)
                
                if sum(appearances) >= 3:  # 최소 3회 출현
                    number_time_series[number] = appearances
            
            # 각 번호별 트렌드 분석
            number_trends = {}
            for number, series in number_time_series.items():
                trend_analysis = self.trend_analyzer.analyze_trend(series)
                if trend_analysis['significance'] > 0.1:
                    number_trends[number] = trend_analysis
            
            result.time_series_data = number_time_series
            result.trends = number_trends
            
            # 전체 트렌드 요약
            if number_trends:
                trending_up = [num for num, trend in number_trends.items() 
                              if trend['direction'] == 'increasing']
                trending_down = [num for num, trend in number_trends.items() 
                                if trend['direction'] == 'decreasing']
                
                result.insights.append(f"상승 트렌드 번호: {len(trending_up)}개")
                result.insights.append(f"하락 트렌드 번호: {len(trending_down)}개")
                
                if trending_up:
                    result.insights.append(f"주요 상승 번호: {trending_up[:5]}")
                if trending_down:
                    result.insights.append(f"주요 하락 번호: {trending_down[:5]}")
        
        except Exception as e:
            result.insights.append(f"번호 시계열 분석 오류: {e}")
        
        return result
    
    def _analyze_aggregate_time_series(self, df: pd.DataFrame) -> TimeSeriesResult:
        """집계 지표의 시계열 분석"""
        result = TimeSeriesResult('aggregate_trends')
        
        try:
            number_cols = [f'num{i}' for i in range(1, 7)]
            
            # 집계 지표들 계산
            aggregate_series = {}
            
            for idx, row in df.iterrows():
                numbers = [row[col] for col in number_cols if pd.notna(row[col])]
                if len(numbers) >= 6:
                    # 다양한 집계 지표
                    aggregate_series.setdefault('sum', []).append(sum(numbers))
                    aggregate_series.setdefault('mean', []).append(np.mean(numbers))
                    aggregate_series.setdefault('std', []).append(np.std(numbers))
                    aggregate_series.setdefault('range', []).append(max(numbers) - min(numbers))
                    aggregate_series.setdefault('odd_count', []).append(sum(1 for n in numbers if n % 2 == 1))
                    
                    # 구간별 분포
                    low_count = sum(1 for n in numbers if n <= 15)
                    mid_count = sum(1 for n in numbers if 16 <= n <= 30)
                    high_count = sum(1 for n in numbers if n >= 31)
                    aggregate_series.setdefault('low_section', []).append(low_count)
                    aggregate_series.setdefault('mid_section', []).append(mid_count)
                    aggregate_series.setdefault('high_section', []).append(high_count)
            
            # 각 집계 지표의 트렌드 분석
            aggregate_trends = {}
            for indicator, series in aggregate_series.items():
                if len(series) >= 10:
                    trend_analysis = self.trend_analyzer.analyze_trend(series)
                    if trend_analysis['significance'] > 0.05:
                        aggregate_trends[indicator] = trend_analysis
            
            result.time_series_data = aggregate_series
            result.trends = aggregate_trends
            
            # 인사이트 생성
            self._generate_aggregate_insights(result, aggregate_trends)
        
        except Exception as e:
            result.insights.append(f"집계 시계열 분석 오류: {e}")
        
        return result
    
    def _generate_aggregate_insights(self, result: TimeSeriesResult, trends: Dict):
        """집계 지표 인사이트 생성"""
        for indicator, trend in trends.items():
            direction = trend['direction']
            strength = trend['strength']
            
            if strength > 0.3:  # 강한 트렌드
                if indicator == 'sum':
                    result.insights.append(f"당첨번호 합계가 {direction} 추세 (강도: {strength:.3f})")
                elif indicator == 'mean':
                    result.insights.append(f"당첨번호 평균이 {direction} 추세")
                elif indicator == 'odd_count':
                    result.insights.append(f"홀수 개수가 {direction} 추세")
                elif indicator in ['low_section', 'mid_section', 'high_section']:
                    section_name = indicator.replace('_section', '구간')
                    result.insights.append(f"{section_name} 번호가 {direction} 추세")


class TrendAnalyzer:
    """트렌드 분석기"""
    
    def analyze_trend(self, series: List[float]) -> Dict[str, Any]:
        """
        시계열의 트렌드 분석
        
        Args:
            series (List[float]): 분석할 시계열 데이터
            
        Returns:
            Dict[str, Any]: 트렌드 분석 결과
        """
        if len(series) < 3:
            return {'direction': 'insufficient_data', 'significance': 0, 'strength': 0}
        
        try:
            # 선형 트렌드 분석
            x = np.arange(len(series))
            slope, intercept, r_value, p_value, std_err = stats.linregress(x, series)
            
            # 트렌드 방향 결정
            if p_value < 0.05:  # 유의한 트렌드
                direction = 'increasing' if slope > 0 else 'decreasing'
                significance = 1 - p_value
            else:
                direction = 'stable'
                significance = 0
            
            # 트렌드 강도 (R²)
            strength = r_value ** 2
            
            # 변화율 계산
            if len(series) > 1:
                total_change = (series[-1] - series[0]) / series[0] if series[0] != 0 else 0
                avg_change_per_period = total_change / len(series)
            else:
                total_change = 0
                avg_change_per_period = 0
            
            # 추가 트렌드 지표
            # 이동평균 트렌드
            if len(series) >= 5:
                window_size = min(5, len(series) // 3)
                moving_avg = pd.Series(series).rolling(window=window_size).mean().dropna()
                ma_slope = (moving_avg.iloc[-1] - moving_avg.iloc[0]) / len(moving_avg) if len(moving_avg) > 1 else 0
            else:
                ma_slope = 0
            
            return {
                'direction': direction,
                'significance': significance,
                'strength': strength,
                'slope': slope,
                'r_squared': r_value ** 2,
                'p_value': p_value,
                'total_change_rate': total_change,
                'avg_change_per_period': avg_change_per_period,
                'moving_average_slope': ma_slope,
                'trend_consistency': self._calculate_trend_consistency(series)
            }
        
        except Exception as e:
            return {
                'direction': 'error',
                'significance': 0,
                'strength': 0,
                'error': str(e)
            }
    
    def _calculate_trend_consistency(self, series: List[float]) -> float:
        """트렌드 일관성 계산"""
        if len(series) < 5:
            return 0
        
        # 연속된 구간들의 트렌드 방향 일관성
        segment_size = max(3, len(series) // 5)
        segment_trends = []
        
        for i in range(0, len(series) - segment_size + 1, segment_size):
            segment = series[i:i + segment_size]
            if len(segment) >= 3:
                x = np.arange(len(segment))
                slope, _, _, p_value, _ = stats.linregress(x, segment)
                if p_value < 0.1:  # 약간 완화된 기준
                    segment_trends.append(1 if slope > 0 else -1)
                else:
                    segment_trends.append(0)  # 무트렌드
        
        if not segment_trends:
            return 0
        
        # 같은 방향 트렌드의 비율
        if len(segment_trends) == 1:
            return 1.0
        
        positive_trends = sum(1 for trend in segment_trends if trend > 0)
        negative_trends = sum(1 for trend in segment_trends if trend < 0)
        max_consistent = max(positive_trends, negative_trends)
        
        return max_consistent / len(segment_trends)


class SeasonalityDetector:
    """계절성 탐지기"""
    
    def detect_seasonality(self, df: pd.DataFrame) -> TimeSeriesResult:
        """계절성 탐지 및 분석"""
        result = TimeSeriesResult('seasonality')
        
        try:
            if 'draw_date' not in df.columns:
                result.insights.append("날짜 정보가 없어 계절성 분석 불가")
                return result
            
            df['draw_date'] = pd.to_datetime(df['draw_date'])
            number_cols = [f'num{i}' for i in range(1, 7)]
            
            # 월별 계절성 분석
            monthly_seasonality = self._analyze_monthly_seasonality(df, number_cols)
            
            # 요일별 계절성 분석
            weekly_seasonality = self._analyze_weekly_seasonality(df, number_cols)
            
            # 분기별 계절성 분석
            quarterly_seasonality = self._analyze_quarterly_seasonality(df, number_cols)
            
            result.seasonal_components = {
                'monthly': monthly_seasonality,
                'weekly': weekly_seasonality,
                'quarterly': quarterly_seasonality
            }
            
            # 계절성 강도 계산
            seasonality_strength = self._calculate_seasonality_strength(
                monthly_seasonality, weekly_seasonality, quarterly_seasonality
            )
            
            result.time_series_data = {
                'seasonality_strength': seasonality_strength,
                'dominant_cycle': self._identify_dominant_cycle(monthly_seasonality, weekly_seasonality)
            }
            
            # 인사이트 생성
            self._generate_seasonality_insights(result, seasonality_strength)
        
        except Exception as e:
            result.insights.append(f"계절성 분석 오류: {e}")
        
        return result
    
    def _analyze_monthly_seasonality(self, df: pd.DataFrame, number_cols: List[str]) -> Dict:
        """월별 계절성 분석"""
        df['month'] = df['draw_date'].dt.month
        monthly_data = {}
        
        for month in range(1, 13):
            month_df = df[df['month'] == month]
            if len(month_df) > 0:
                # 월별 번호 통계
                all_numbers = []
                for col in number_cols:
                    if col in month_df.columns:
                        all_numbers.extend(month_df[col].dropna().astype(int).tolist())
                
                if all_numbers:
                    monthly_data[month] = {
                        'draw_count': len(month_df),
                        'total_numbers': len(all_numbers),
                        'avg_number': np.mean(all_numbers),
                        'std_number': np.std(all_numbers),
                        'number_distribution': Counter(all_numbers).most_common(10),
                        'odd_ratio': sum(1 for n in all_numbers if n % 2 == 1) / len(all_numbers)
                    }
        
        return monthly_data
    
    def _analyze_weekly_seasonality(self, df: pd.DataFrame, number_cols: List[str]) -> Dict:
        """요일별 계절성 분석"""
        df['day_of_week'] = df['draw_date'].dt.dayofweek
        weekly_data = {}
        
        for dow in range(7):
            dow_df = df[df['day_of_week'] == dow]
            if len(dow_df) > 2:
                # 요일별 번호 통계
                all_numbers = []
                for col in number_cols:
                    if col in dow_df.columns:
                        all_numbers.extend(dow_df[col].dropna().astype(int).tolist())
                
                if all_numbers:
                    weekly_data[dow] = {
                        'draw_count': len(dow_df),
                        'avg_number': np.mean(all_numbers),
                        'std_number': np.std(all_numbers),
                        'number_distribution': Counter(all_numbers).most_common(5)
                    }
        
        return weekly_data
    
    def _analyze_quarterly_seasonality(self, df: pd.DataFrame, number_cols: List[str]) -> Dict:
        """분기별 계절성 분석"""
        df['quarter'] = df['draw_date'].dt.quarter
        quarterly_data = {}
        
        for quarter in range(1, 5):
            quarter_df = df[df['quarter'] == quarter]
            if len(quarter_df) > 5:
                # 분기별 번호 통계
                all_numbers = []
                for col in number_cols:
                    if col in quarter_df.columns:
                        all_numbers.extend(quarter_df[col].dropna().astype(int).tolist())
                
                if all_numbers:
                    quarterly_data[quarter] = {
                        'draw_count': len(quarter_df),
                        'avg_number': np.mean(all_numbers),
                        'std_number': np.std(all_numbers),
                        'number_distribution': Counter(all_numbers).most_common(10)
                    }
        
        return quarterly_data
    
    def _calculate_seasonality_strength(self, monthly: Dict, weekly: Dict, quarterly: Dict) -> Dict:
        """계절성 강도 계산"""
        strength = {}
        
        # 월별 계절성 강도
        if monthly and len(monthly) >= 6:
            monthly_avgs = [data['avg_number'] for data in monthly.values()]
            monthly_strength = np.std(monthly_avgs) / np.mean(monthly_avgs) if np.mean(monthly_avgs) > 0 else 0
            strength['monthly'] = monthly_strength
        
        # 요일별 계절성 강도
        if weekly and len(weekly) >= 3:
            weekly_avgs = [data['avg_number'] for data in weekly.values()]
            weekly_strength = np.std(weekly_avgs) / np.mean(weekly_avgs) if np.mean(weekly_avgs) > 0 else 0
            strength['weekly'] = weekly_strength
        
        # 분기별 계절성 강도
        if quarterly and len(quarterly) >= 3:
            quarterly_avgs = [data['avg_number'] for data in quarterly.values()]
            quarterly_strength = np.std(quarterly_avgs) / np.mean(quarterly_avgs) if np.mean(quarterly_avgs) > 0 else 0
            strength['quarterly'] = quarterly_strength
        
        return strength
    
    def _identify_dominant_cycle(self, monthly: Dict, weekly: Dict) -> str:
        """주도적인 주기 식별"""
        monthly_strength = 0
        weekly_strength = 0
        
        if monthly and len(monthly) >= 6:
            monthly_avgs = [data['avg_number'] for data in monthly.values()]
            monthly_strength = np.std(monthly_avgs) / np.mean(monthly_avgs) if np.mean(monthly_avgs) > 0 else 0
        
        if weekly and len(weekly) >= 3:
            weekly_avgs = [data['avg_number'] for data in weekly.values()]
            weekly_strength = np.std(weekly_avgs) / np.mean(weekly_avgs) if np.mean(weekly_avgs) > 0 else 0
        
        if monthly_strength > weekly_strength and monthly_strength > 0.05:
            return 'monthly'
        elif weekly_strength > 0.05:
            return 'weekly'
        else:
            return 'none'
    
    def _generate_seasonality_insights(self, result: TimeSeriesResult, strength: Dict):
        """계절성 인사이트 생성"""
        for cycle_type, cycle_strength in strength.items():
            if cycle_strength > 0.1:
                result.insights.append(f"{cycle_type} 계절성 감지 (강도: {cycle_strength:.3f})")


class CyclicalPatternAnalyzer:
    """순환 패턴 분석기"""
    
    def analyze_cycles(self, df: pd.DataFrame) -> TimeSeriesResult:
        """순환 패턴 분석"""
        result = TimeSeriesResult('cyclical_patterns')
        
        try:
            number_cols = [f'num{i}' for i in range(1, 7)]
            
            # 번호별 순환 패턴 분석
            cyclical_patterns = {}
            
            for number in range(1, 46):
                # 해당 번호의 출현 간격 분석
                appearances = []
                for idx, row in df.iterrows():
                    if any(row[col] == number for col in number_cols if pd.notna(row[col])):
                        appearances.append(idx)
                
                if len(appearances) >= 4:  # 최소 4회 출현
                    gaps = [appearances[i+1] - appearances[i] for i in range(len(appearances)-1)]
                    cycle_analysis = self._analyze_cycle_pattern(gaps)
                    
                    if cycle_analysis['cycle_strength'] > 0.3:
                        cyclical_patterns[number] = cycle_analysis
            
            # 전체 시스템의 순환성 분석
            system_cycles = self._analyze_system_cycles(df, number_cols)
            
            result.time_series_data = {
                'individual_cycles': cyclical_patterns,
                'system_cycles': system_cycles
            }
            
            # 인사이트 생성
            if cyclical_patterns:
                result.insights.append(f"순환 패턴을 보이는 번호: {len(cyclical_patterns)}개")
                strong_cycles = [num for num, data in cyclical_patterns.items() 
                               if data['cycle_strength'] > 0.5]
                if strong_cycles:
                    result.insights.append(f"강한 순환 패턴 번호: {strong_cycles[:5]}")
        
        except Exception as e:
            result.insights.append(f"순환 패턴 분석 오류: {e}")
        
        return result
    
    def _analyze_cycle_pattern(self, gaps: List[int]) -> Dict:
        """개별 번호의 순환 패턴 분석"""
        if len(gaps) < 3:
            return {'cycle_strength': 0}
        
        # 간격의 일관성 측정
        gap_mean = np.mean(gaps)
        gap_std = np.std(gaps)
        consistency = 1 - (gap_std / gap_mean) if gap_mean > 0 else 0
        
        # 주기성 검사 (FFT 사용)
        cycle_strength = 0
        dominant_period = 0
        
        if len(gaps) >= 8:
            # FFT로 주기성 탐지
            fft_result = fft(gaps)
            frequencies = fftfreq(len(gaps))
            
            # 0주파수 제외하고 가장 강한 주파수 찾기
            power_spectrum = np.abs(fft_result[1:len(gaps)//2])
            if len(power_spectrum) > 0:
                max_power_idx = np.argmax(power_spectrum)
                cycle_strength = power_spectrum[max_power_idx] / np.sum(power_spectrum)
                
                if frequencies[max_power_idx + 1] != 0:
                    dominant_period = 1 / abs(frequencies[max_power_idx + 1])
        
        return {
            'cycle_strength': min(consistency, cycle_strength),
            'average_gap': gap_mean,
            'gap_consistency': consistency,
            'dominant_period': dominant_period,
            'total_appearances': len(gaps) + 1
        }
    
    def _analyze_system_cycles(self, df: pd.DataFrame, number_cols: List[str]) -> Dict:
        """전체 시스템의 순환성 분석"""
        system_indicators = []
        
        # 시스템 지표 시계열 생성
        for idx, row in df.iterrows():
            numbers = [row[col] for col in number_cols if pd.notna(row[col])]
            if len(numbers) >= 6:
                # 다양한 시스템 지표
                indicators = {
                    'sum': sum(numbers),
                    'mean': np.mean(numbers),
                    'range': max(numbers) - min(numbers),
                    'odd_count': sum(1 for n in numbers if n % 2 == 1)
                }
                system_indicators.append(indicators)
        
        # 각 지표의 순환성 분석
        system_cycles = {}
        for indicator in ['sum', 'mean', 'range', 'odd_count']:
            series = [item[indicator] for item in system_indicators]
            if len(series) >= 10:
                cycle_analysis = self._detect_cycles_fft(series)
                if cycle_analysis['strength'] > 0.1:
                    system_cycles[indicator] = cycle_analysis
        
        return system_cycles
    
    def _detect_cycles_fft(self, series: List[float]) -> Dict:
        """FFT를 사용한 주기 탐지"""
        if len(series) < 8:
            return {'strength': 0, 'period': 0}
        
        try:
            # 노이즈 제거를 위한 스무딩
            if len(series) >= 5:
                window_size = min(5, len(series) // 4)
                if window_size >= 3 and window_size % 2 == 0:
                    window_size += 1  # 홀수로 만들기
                if window_size >= 3:
                    smoothed = savgol_filter(series, window_size, 2)
                else:
                    smoothed = series
            else:
                smoothed = series
            
            # FFT 분석
            fft_result = fft(smoothed)
            frequencies = fftfreq(len(smoothed))
            
            # 파워 스펙트럼 계산
            power_spectrum = np.abs(fft_result[1:len(smoothed)//2])
            
            if len(power_spectrum) > 0:
                # 가장 강한 주파수 찾기
                max_power_idx = np.argmax(power_spectrum)
                max_power = power_spectrum[max_power_idx]
                total_power = np.sum(power_spectrum)
                
                strength = max_power / total_power if total_power > 0 else 0
                
                # 주기 계산
                freq = frequencies[max_power_idx + 1]
                period = 1 / abs(freq) if freq != 0 else 0
                
                return {
                    'strength': strength,
                    'period': period,
                    'dominant_frequency': freq,
                    'power_ratio': strength
                }
        
        except Exception:
            pass
        
        return {'strength': 0, 'period': 0}


class ForecastEngine:
    """예측 엔진"""
    
    def generate_forecasts(self, df: pd.DataFrame, horizon: int) -> TimeSeriesResult:
        """시계열 예측 생성"""
        result = TimeSeriesResult('forecasts')
        
        try:
            number_cols = [f'num{i}' for i in range(1, 7)]
            
            # 집계 지표 예측
            aggregate_forecasts = self._forecast_aggregates(df, number_cols, horizon)
            
            # 개별 번호 출현 확률 예측
            number_probability_forecasts = self._forecast_number_probabilities(df, number_cols, horizon)
            
            result.forecasts = {
                'aggregate_indicators': aggregate_forecasts,
                'number_probabilities': number_probability_forecasts
            }
            
            # 신뢰구간 계산
            result.confidence_intervals = self._calculate_confidence_intervals(
                aggregate_forecasts, horizon
            )
            
            # 예측 품질 평가
            result.model_performance = self._evaluate_forecast_quality(df, number_cols)
            
            # 인사이트 생성
            self._generate_forecast_insights(result)
        
        except Exception as e:
            result.insights.append(f"예측 생성 오류: {e}")
        
        return result
    
    def _forecast_aggregates(self, df: pd.DataFrame, number_cols: List[str], horizon: int) -> Dict:
        """집계 지표 예측"""
        forecasts = {}
        
        # 시계열 데이터 준비
        aggregate_series = {}
        for idx, row in df.iterrows():
            numbers = [row[col] for col in number_cols if pd.notna(row[col])]
            if len(numbers) >= 6:
                aggregate_series.setdefault('sum', []).append(sum(numbers))
                aggregate_series.setdefault('mean', []).append(np.mean(numbers))
                aggregate_series.setdefault('odd_count', []).append(sum(1 for n in numbers if n % 2 == 1))
        
        # 각 지표에 대해 단순 예측 수행
        for indicator, series in aggregate_series.items():
            if len(series) >= 5:
                forecast = self._simple_forecast(series, horizon)
                forecasts[indicator] = forecast
        
        return forecasts
    
    def _simple_forecast(self, series: List[float], horizon: int) -> Dict:
        """단순 예측 모델"""
        if len(series) < 3:
            return {'values': [np.mean(series)] * horizon, 'method': 'mean'}
        
        # 이동평균 예측
        window_size = min(5, len(series) // 2)
        recent_avg = np.mean(series[-window_size:])
        
        # 트렌드 고려
        if len(series) >= 5:
            x = np.arange(len(series))
            slope, intercept, _, p_value, _ = stats.linregress(x, series)
            
            if p_value < 0.1:  # 유의한 트렌드
                # 트렌드 연장 예측
                forecast_values = []
                for h in range(1, horizon + 1):
                    forecast_val = series[-1] + slope * h
                    forecast_values.append(forecast_val)
                
                return {
                    'values': forecast_values,
                    'method': 'linear_trend',
                    'slope': slope,
                    'base_value': series[-1]
                }
        
        # 트렌드가 없으면 이동평균
        return {
            'values': [recent_avg] * horizon,
            'method': 'moving_average',
            'window_size': window_size
        }
    
    def _forecast_number_probabilities(self, df: pd.DataFrame, number_cols: List[str], horizon: int) -> Dict:
        """번호별 출현 확률 예측"""
        # 최근 가중치를 적용한 확률 계산
        recent_weight = 0.3  # 최근 30% 가중치
        
        number_probabilities = {}
        total_draws = len(df)
        
        for number in range(1, 46):
            # 전체 출현 빈도
            total_appearances = 0
            recent_appearances = 0
            recent_window = min(20, total_draws // 4)
            
            for idx, row in df.iterrows():
                is_present = any(row[col] == number for col in number_cols if pd.notna(row[col]))
                if is_present:
                    total_appearances += 1
                    if idx >= total_draws - recent_window:
                        recent_appearances += 1
            
            # 가중 확률 계산
            overall_prob = total_appearances / (total_draws * 6) if total_draws > 0 else 0
            recent_prob = recent_appearances / (recent_window * 6) if recent_window > 0 else 0
            
            weighted_prob = overall_prob * (1 - recent_weight) + recent_prob * recent_weight
            number_probabilities[number] = weighted_prob
        
        return number_probabilities
    
    def _calculate_confidence_intervals(self, forecasts: Dict, horizon: int) -> Dict:
        """신뢰구간 계산"""
        confidence_intervals = {}
        
        for indicator, forecast_data in forecasts.items():
            if 'values' in forecast_data:
                values = forecast_data['values']
                
                # 단순 신뢰구간 (예측값의 ±20%)
                lower_bounds = [v * 0.8 for v in values]
                upper_bounds = [v * 1.2 for v in values]
                
                confidence_intervals[indicator] = {
                    'lower_80': lower_bounds,
                    'upper_80': upper_bounds,
                    'point_forecast': values
                }
        
        return confidence_intervals
    
    def _evaluate_forecast_quality(self, df: pd.DataFrame, number_cols: List[str]) -> Dict:
        """예측 품질 평가 (백테스팅)"""
        if len(df) < 20:
            return {'message': '백테스팅을 위한 데이터 부족'}
        
        # 데이터를 훈련/테스트로 분할
        split_point = len(df) - 10
        train_df = df.iloc[:split_point]
        test_df = df.iloc[split_point:]
        
        # 훈련 데이터로 예측 수행
        train_forecast = self._forecast_aggregates(train_df, number_cols, len(test_df))
        
        # 실제값과 비교
        test_actuals = {}
        for idx, row in test_df.iterrows():
            numbers = [row[col] for col in number_cols if pd.notna(row[col])]
            if len(numbers) >= 6:
                test_actuals.setdefault('sum', []).append(sum(numbers))
                test_actuals.setdefault('mean', []).append(np.mean(numbers))
                test_actuals.setdefault('odd_count', []).append(sum(1 for n in numbers if n % 2 == 1))
        
        # 오차 계산
        performance = {}
        for indicator in ['sum', 'mean', 'odd_count']:
            if indicator in train_forecast and indicator in test_actuals:
                predicted = train_forecast[indicator]['values']
                actual = test_actuals[indicator]
                
                min_len = min(len(predicted), len(actual))
                if min_len > 0:
                    pred_subset = predicted[:min_len]
                    actual_subset = actual[:min_len]
                    
                    mae = np.mean(np.abs(np.array(pred_subset) - np.array(actual_subset)))
                    mape = np.mean(np.abs((np.array(actual_subset) - np.array(pred_subset)) / np.array(actual_subset))) * 100
                    
                    performance[indicator] = {
                        'mae': mae,
                        'mape': mape,
                        'accuracy': max(0, 100 - mape)
                    }
        
        return performance
    
    def _generate_forecast_insights(self, result: TimeSeriesResult):
        """예측 인사이트 생성"""
        performance = result.model_performance
        
        if performance:
            # 예측 정확도 분석
            accuracies = [perf['accuracy'] for perf in performance.values() if 'accuracy' in perf]
            if accuracies:
                avg_accuracy = np.mean(accuracies)
                result.insights.append(f"평균 예측 정확도: {avg_accuracy:.1f}%")
                
                if avg_accuracy > 70:
                    result.insights.append("예측 모델의 성능이 우수함")
                elif avg_accuracy < 50:
                    result.insights.append("예측 모델의 성능이 제한적임")
        
        # 예측 트렌드 분석
        forecasts = result.forecasts.get('aggregate_indicators', {})
        for indicator, forecast_data in forecasts.items():
            if 'values' in forecast_data and len(forecast_data['values']) > 1:
                values = forecast_data['values']
                if values[-1] > values[0]:
                    result.insights.append(f"{indicator} 지표가 상승 예측됨")
                elif values[-1] < values[0]:
                    result.insights.append(f"{indicator} 지표가 하락 예측됨")


# === 모듈 테스트 ===
if __name__ == "__main__":
    # 테스트용 데이터 생성
    np.random.seed(42)
    
    test_data = []
    base_date = datetime(2020, 1, 1)
    
    for i in range(100):
        # 시간에 따른 패턴을 의도적으로 삽입
        time_factor = i / 100
        
        # 시간에 따라 평균이 증가하는 트렌드
        if i < 30:
            base_numbers = np.random.choice(range(1, 25), size=6, replace=False)
        elif i < 70:
            base_numbers = np.random.choice(range(10, 35), size=6, replace=False)
        else:
            base_numbers = np.random.choice(range(20, 46), size=6, replace=False)
        
        # 월별 계절성 추가
        month = (base_date + timedelta(days=i*7)).month
        if month in [12, 1, 2]:  # 겨울
            seasonal_bias = [-2, -1, 0, 1, 2, 3]
        elif month in [6, 7, 8]:  # 여름
            seasonal_bias = [2, 3, 1, 0, -1, -2]
        else:
            seasonal_bias = [0, 0, 0, 0, 0, 0]
        
        # 계절 편향 적용
        numbers = []
        for j, num in enumerate(sorted(base_numbers)):
            adjusted = num + seasonal_bias[j]
            adjusted = max(1, min(45, adjusted))  # 범위 제한
            numbers.append(adjusted)
        
        # 중복 제거
        numbers = list(set(numbers))
        while len(numbers) < 6:
            new_num = np.random.randint(1, 46)
            if new_num not in numbers:
                numbers.append(new_num)
        
        numbers = sorted(numbers[:6])
        
        row = {
            'round': 1000 + i,
            'draw_date': base_date + timedelta(days=i*7),  # 주간 간격
        }
        
        for j, num in enumerate(numbers, 1):
            row[f'num{j}'] = num
        
        test_data.append(row)
    
    test_df = pd.DataFrame(test_data)
    
    print("=== 시계열 분석기 테스트 ===")
    print(f"테스트 데이터: {len(test_df)}행")
    print(f"기간: {test_df['draw_date'].min()} ~ {test_df['draw_date'].max()}")
    
    try:
        # 시계열 분석기 초기화
        analyzer = LottoTimeSeriesAnalyzer(forecast_horizon=5)
        
        # 전체 분석 수행
        results = analyzer.analyze_time_series(test_df)
        
        print(f"\n📈 시계열 분석 결과:")
        
        for analysis_type, result in results.items():
            print(f"\n{analysis_type.upper()}:")
            
            if result.insights:
                print("  주요 인사이트:")
                for insight in result.insights[:3]:
                    print(f"    - {insight}")
            
            if result.trends:
                print("  트렌드 정보:")
                for trend_name, trend_data in list(result.trends.items())[:3]:
                    if isinstance(trend_data, dict) and 'direction' in trend_data:
                        direction = trend_data['direction']
                        strength = trend_data.get('strength', 0)
                        print(f"    {trend_name}: {direction} (강도: {strength:.3f})")
            
            if result.forecasts:
                print("  예측 정보:")
                for forecast_name, forecast_data in list(result.forecasts.items())[:2]:
                    print(f"    {forecast_name}: 사용 가능")
        
        # 개별 분석기 테스트
        print(f"\n🔍 개별 분석기 테스트:")
        
        # 트렌드 분석기
        test_series = [10, 12, 11, 13, 15, 14, 16, 18, 17, 19]
        trend_result = analyzer.trend_analyzer.analyze_trend(test_series)
        print(f"  트렌드 분석: 방향 {trend_result['direction']}, 강도 {trend_result['strength']:.3f}")
        
        # 계절성 탐지기
        seasonality_result = analyzer.seasonality_detector.detect_seasonality(test_df)
        print(f"  계절성 탐지: 인사이트 {len(seasonality_result.insights)}개")
        
        # 순환 패턴 분석기
        cyclical_result = analyzer.cyclical_analyzer.analyze_cycles(test_df)
        print(f"  순환 패턴: 인사이트 {len(cyclical_result.insights)}개")
        
        # 예측 엔진
        forecast_result = analyzer.forecast_engine.generate_forecasts(test_df, 3)
        print(f"  예측 엔진: 인사이트 {len(forecast_result.insights)}개")
        
        if forecast_result.model_performance:
            print("  예측 성능:")
            for indicator, perf in forecast_result.model_performance.items():
                if 'accuracy' in perf:
                    print(f"    {indicator}: {perf['accuracy']:.1f}% 정확도")
        
        print("\n✅ 시계열 분석기 테스트 완료")
        
    except Exception as e:
        print(f"❌ 테스트 중 오류: {e}")
        import traceback
        traceback.print_exc()
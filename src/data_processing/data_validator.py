"""
파일명: src/data_processing/data_validator.py
목적: 로또 데이터 검증 및 무결성 확인
작성자: AI Assistant
작성일: 2025-08-31
버전: 1.0

주요 클래스/함수:
- LottoDataValidator: 로또 데이터 검증 클래스
- validate_winning_numbers: 당첨번호 검증
- validate_data_consistency: 데이터 일관성 검증
- validate_business_rules: 비즈니스 룰 검증
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any, Optional, Set
from collections import Counter
import logging
from datetime import datetime, timedelta
from pathlib import Path
import sys
import warnings

# 상위 디렉토리 imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.utils.config import Config, LottoConstants

warnings.filterwarnings('ignore')

class ValidationResult:
    """검증 결과를 담는 클래스"""
    
    def __init__(self):
        self.is_valid = True
        self.errors = []
        self.warnings = []
        self.info = []
        self.statistics = {}
    
    def add_error(self, message: str, row_index: Optional[int] = None):
        """오류 추가"""
        self.is_valid = False
        error_msg = f"행 {row_index}: {message}" if row_index is not None else message
        self.errors.append(error_msg)
    
    def add_warning(self, message: str, row_index: Optional[int] = None):
        """경고 추가"""
        warning_msg = f"행 {row_index}: {message}" if row_index is not None else message
        self.warnings.append(warning_msg)
    
    def add_info(self, message: str):
        """정보 추가"""
        self.info.append(message)
    
    def get_summary(self) -> Dict[str, Any]:
        """검증 결과 요약"""
        return {
            'is_valid': self.is_valid,
            'error_count': len(self.errors),
            'warning_count': len(self.warnings),
            'info_count': len(self.info),
            'errors': self.errors,
            'warnings': self.warnings,
            'info': self.info,
            'statistics': self.statistics
        }


class LottoDataValidator:
    """로또 데이터 검증 클래스"""
    
    def __init__(self, tolerance_level: str = 'medium'):
        """
        초기화
        
        Args:
            tolerance_level (str): 허용 수준 ('strict', 'medium', 'loose')
        """
        self.tolerance_level = tolerance_level
        self.logger = self._setup_logger()
        
        # 허용 수준별 임계값 설정
        self.thresholds = {
            'strict': {
                'missing_data_ratio': 0.01,      # 1% 미만
                'outlier_ratio': 0.005,          # 0.5% 미만
                'duplicate_ratio': 0.0,          # 중복 허용 안함
                'date_gap_days': 7,              # 정확히 7일
                'number_frequency_deviation': 0.1 # 10% 편차
            },
            'medium': {
                'missing_data_ratio': 0.05,      # 5% 미만
                'outlier_ratio': 0.02,           # 2% 미만
                'duplicate_ratio': 0.001,        # 0.1% 미만
                'date_gap_days': 2,              # ±2일
                'number_frequency_deviation': 0.2 # 20% 편차
            },
            'loose': {
                'missing_data_ratio': 0.1,       # 10% 미만
                'outlier_ratio': 0.05,           # 5% 미만
                'duplicate_ratio': 0.01,         # 1% 미만
                'date_gap_days': 5,              # ±5일
                'number_frequency_deviation': 0.3 # 30% 편차
            }
        }
        
        self.current_thresholds = self.thresholds[tolerance_level]
        
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
    
    def validate_data(self, df: pd.DataFrame) -> ValidationResult:
        """
        전체 데이터 검증
        
        Args:
            df (pd.DataFrame): 검증할 데이터프레임
            
        Returns:
            ValidationResult: 검증 결과
        """
        self.logger.info(f"데이터 검증 시작 (허용 수준: {self.tolerance_level})")
        result = ValidationResult()
        
        try:
            # 1. 기본 구조 검증
            self._validate_basic_structure(df, result)
            
            # 2. 당첨번호 검증
            self._validate_winning_numbers(df, result)
            
            # 3. 날짜 검증
            self._validate_dates(df, result)
            
            # 4. 당첨금액 검증
            self._validate_prize_amounts(df, result)
            
            # 5. 당첨자 수 검증
            self._validate_winner_counts(df, result)
            
            # 6. 데이터 일관성 검증
            self._validate_data_consistency(df, result)
            
            # 7. 비즈니스 룰 검증
            self._validate_business_rules(df, result)
            
            # 8. 통계적 검증
            self._validate_statistical_properties(df, result)
            
            # 9. 최종 통계 생성
            self._generate_statistics(df, result)
            
            status = "통과" if result.is_valid else "실패"
            self.logger.info(f"데이터 검증 완료: {status} (오류: {len(result.errors)}, 경고: {len(result.warnings)})")
            
            return result
            
        except Exception as e:
            self.logger.error(f"데이터 검증 중 오류: {e}")
            result.add_error(f"검증 프로세스 오류: {e}")
            return result
    
    def _validate_basic_structure(self, df: pd.DataFrame, result: ValidationResult):
        """기본 구조 검증"""
        self.logger.info("기본 구조 검증")
        
        # 데이터프레임 비어있음 확인
        if df.empty:
            result.add_error("데이터프레임이 비어있습니다.")
            return
        
        # 필수 컬럼 확인
        required_columns = ['num1', 'num2', 'num3', 'num4', 'num5', 'num6']
        missing_columns = [col for col in required_columns if col not in df.columns]
        
        if missing_columns:
            result.add_error(f"필수 컬럼이 누락되었습니다: {missing_columns}")
        
        # 컬럼 수 확인
        if len(df.columns) < 6:
            result.add_error(f"컬럼 수가 부족합니다. (현재: {len(df.columns)}, 최소: 6)")
        
        # 행 수 확인
        if len(df) < 10:
            result.add_warning(f"데이터 행 수가 적습니다. (현재: {len(df)})")
        
        result.add_info(f"데이터 크기: {len(df)} 행 × {len(df.columns)} 열")
    
    def _validate_winning_numbers(self, df: pd.DataFrame, result: ValidationResult):
        """당첨번호 검증"""
        self.logger.info("당첨번호 검증")
        
        number_columns = [col for col in df.columns if col in ['num1', 'num2', 'num3', 'num4', 'num5', 'num6', 'bonus']]
        
        for idx, row in df.iterrows():
            # 각 번호가 1-45 범위에 있는지 확인
            for col in number_columns:
                if col in df.columns and pd.notna(row[col]):
                    number = row[col]
                    if not (1 <= number <= 45):
                        result.add_error(f"번호 범위 오류: {col}={number} (1-45 범위 외)", idx)
            
            # 메인 당첨번호 6개가 모두 다른지 확인
            main_numbers = []
            for col in ['num1', 'num2', 'num3', 'num4', 'num5', 'num6']:
                if col in df.columns and pd.notna(row[col]):
                    main_numbers.append(row[col])
            
            if len(main_numbers) >= 6:
                if len(main_numbers) != len(set(main_numbers)):
                    result.add_error(f"중복 당첨번호 발견: {main_numbers}", idx)
                
                # 정렬 순서 확인
                if main_numbers != sorted(main_numbers):
                    result.add_warning(f"당첨번호가 정렬되지 않음: {main_numbers}", idx)
            
            # 보너스 번호가 메인 번호와 중복되지 않는지 확인
            if 'bonus' in df.columns and pd.notna(row['bonus']):
                bonus = row['bonus']
                if bonus in main_numbers:
                    result.add_error(f"보너스 번호가 메인 번호와 중복: {bonus}", idx)
        
        # 결측치 확인
        for col in number_columns:
            if col in df.columns:
                missing_count = df[col].isna().sum()
                missing_ratio = missing_count / len(df)
                
                if missing_ratio > self.current_thresholds['missing_data_ratio']:
                    result.add_error(f"{col}: 결측치 비율이 높음 ({missing_ratio:.2%})")
                elif missing_count > 0:
                    result.add_warning(f"{col}: 결측치 {missing_count}개")
    
    def _validate_dates(self, df: pd.DataFrame, result: ValidationResult):
        """날짜 검증"""
        self.logger.info("날짜 검증")
        
        date_columns = [col for col in df.columns if 'date' in col.lower() or '날짜' in col.lower()]
        
        for col in date_columns:
            if col in df.columns:
                # 날짜 형식 확인
                try:
                    date_series = pd.to_datetime(df[col], errors='coerce')
                    invalid_dates = date_series.isna().sum()
                    
                    if invalid_dates > 0:
                        result.add_warning(f"{col}: 유효하지 않은 날짜 {invalid_dates}개")
                    
                    valid_dates = date_series.dropna()
                    if len(valid_dates) > 1:
                        # 날짜 간격 확인 (로또는 보통 주 2회)
                        date_diffs = valid_dates.diff().dropna()
                        
                        # 3일 또는 4일 간격이 일반적
                        expected_gaps = [3, 4, 7]  # 3-4일 (주 2회), 7일 (주 1회)
                        
                        for i, diff in enumerate(date_diffs):
                            days_diff = diff.days
                            if not any(abs(days_diff - gap) <= self.current_thresholds['date_gap_days'] 
                                     for gap in expected_gaps):
                                if days_diff > 30:  # 30일 초과는 오류
                                    result.add_error(f"{col}: 비정상적인 날짜 간격 ({days_diff}일)", i+1)
                                else:
                                    result.add_warning(f"{col}: 예상과 다른 날짜 간격 ({days_diff}일)", i+1)
                        
                        # 미래 날짜 확인
                        future_dates = valid_dates > pd.Timestamp.now()
                        if future_dates.sum() > 0:
                            result.add_warning(f"{col}: 미래 날짜 {future_dates.sum()}개")
                        
                        # 너무 과거 날짜 확인 (1945년 이전)
                        too_old = valid_dates < pd.Timestamp('1945-01-01')
                        if too_old.sum() > 0:
                            result.add_warning(f"{col}: 너무 과거 날짜 {too_old.sum()}개")
                
                except Exception as e:
                    result.add_error(f"{col}: 날짜 처리 오류 - {e}")
    
    def _validate_prize_amounts(self, df: pd.DataFrame, result: ValidationResult):
        """당첨금액 검증"""
        self.logger.info("당첨금액 검증")
        
        prize_columns = [col for col in df.columns if 'prize' in col.lower() or '금액' in col.lower()]
        
        for col in prize_columns:
            if col in df.columns:
                # 숫자 변환 가능 확인
                numeric_series = pd.to_numeric(df[col], errors='coerce')
                invalid_amounts = numeric_series.isna().sum()
                
                if invalid_amounts > 0:
                    result.add_warning(f"{col}: 숫자로 변환 불가능한 값 {invalid_amounts}개")
                
                valid_amounts = numeric_series.dropna()
                if len(valid_amounts) > 0:
                    # 음수 값 확인
                    negative_count = (valid_amounts < 0).sum()
                    if negative_count > 0:
                        result.add_error(f"{col}: 음수 값 {negative_count}개")
                    
                    # 비현실적으로 큰 값 확인 (1조원 초과)
                    too_large_count = (valid_amounts > 1_000_000_000_000).sum()
                    if too_large_count > 0:
                        result.add_warning(f"{col}: 비현실적으로 큰 금액 {too_large_count}개")
                    
                    # 비현실적으로 작은 값 확인 (1000원 미만)
                    too_small_count = ((valid_amounts > 0) & (valid_amounts < 1000)).sum()
                    if too_small_count > 0:
                        result.add_warning(f"{col}: 비현실적으로 작은 금액 {too_small_count}개")
                    
                    # 통계적 이상치 확인
                    if len(valid_amounts) > 10:
                        q1 = valid_amounts.quantile(0.25)
                        q3 = valid_amounts.quantile(0.75)
                        iqr = q3 - q1
                        lower_bound = q1 - 3 * iqr
                        upper_bound = q3 + 3 * iqr
                        
                        outliers = ((valid_amounts < lower_bound) | (valid_amounts > upper_bound)).sum()
                        outlier_ratio = outliers / len(valid_amounts)
                        
                        if outlier_ratio > self.current_thresholds['outlier_ratio']:
                            result.add_warning(f"{col}: 이상치 비율이 높음 ({outlier_ratio:.2%})")
    
    def _validate_winner_counts(self, df: pd.DataFrame, result: ValidationResult):
        """당첨자 수 검증"""
        self.logger.info("당첨자 수 검증")
        
        winner_columns = [col for col in df.columns if 'winner' in col.lower() or '당첨자' in col.lower()]
        
        for col in winner_columns:
            if col in df.columns:
                # 숫자 변환 확인
                numeric_series = pd.to_numeric(df[col], errors='coerce')
                invalid_counts = numeric_series.isna().sum()
                
                if invalid_counts > 0:
                    result.add_warning(f"{col}: 숫자로 변환 불가능한 값 {invalid_counts}개")
                
                valid_counts = numeric_series.dropna()
                if len(valid_counts) > 0:
                    # 음수 값 확인
                    negative_count = (valid_counts < 0).sum()
                    if negative_count > 0:
                        result.add_error(f"{col}: 음수 당첨자 수 {negative_count}개")
                    
                    # 정수 여부 확인
                    non_integer = (valid_counts != valid_counts.astype(int)).sum()
                    if non_integer > 0:
                        result.add_warning(f"{col}: 정수가 아닌 당첨자 수 {non_integer}개")
                    
                    # 비현실적으로 큰 값 확인 (1억 명 초과)
                    too_large = (valid_counts > 100_000_000).sum()
                    if too_large > 0:
                        result.add_error(f"{col}: 비현실적으로 많은 당첨자 {too_large}개")
    
    def _validate_data_consistency(self, df: pd.DataFrame, result: ValidationResult):
        """데이터 일관성 검증"""
        self.logger.info("데이터 일관성 검증")
        
        # 중복 행 확인
        duplicate_rows = df.duplicated().sum()
        duplicate_ratio = duplicate_rows / len(df)
        
        if duplicate_ratio > self.current_thresholds['duplicate_ratio']:
            result.add_error(f"중복 행 비율이 높음: {duplicate_ratio:.2%} ({duplicate_rows}개)")
        elif duplicate_rows > 0:
            result.add_warning(f"중복 행 {duplicate_rows}개 발견")
        
        # 당첨금액과 당첨자 수의 관계 확인
        if '1st_prize' in df.columns and '1st_winners' in df.columns:
            prize_series = pd.to_numeric(df['1st_prize'], errors='coerce')
            winner_series = pd.to_numeric(df['1st_winners'], errors='coerce')
            
            valid_mask = ~(prize_series.isna() | winner_series.isna())
            valid_prize = prize_series[valid_mask]
            valid_winners = winner_series[valid_mask]
            
            if len(valid_prize) > 0:
                # 당첨자가 0명인데 당첨금이 있는 경우
                zero_winners_with_prize = ((valid_winners == 0) & (valid_prize > 0)).sum()
                if zero_winners_with_prize > 0:
                    result.add_warning(f"당첨자 0명이지만 당첨금이 있는 경우: {zero_winners_with_prize}개")
                
                # 당첨자가 있는데 당첨금이 0인 경우
                winners_without_prize = ((valid_winners > 0) & (valid_prize == 0)).sum()
                if winners_without_prize > 0:
                    result.add_warning(f"당첨자가 있지만 당첨금이 0인 경우: {winners_without_prize}개")
        
        # 회차 번호 연속성 확인
        if 'round' in df.columns:
            rounds = pd.to_numeric(df['round'], errors='coerce').dropna()
            if len(rounds) > 1:
                sorted_rounds = rounds.sort_values()
                gaps = sorted_rounds.diff().dropna()
                
                # 회차는 보통 1씩 증가
                non_sequential = (gaps != 1).sum()
                if non_sequential > len(rounds) * 0.1:  # 10% 이상이 비순차적
                    result.add_warning(f"비순차적 회차 번호: {non_sequential}개")
    
    def _validate_business_rules(self, df: pd.DataFrame, result: ValidationResult):
        """비즈니스 룰 검증"""
        self.logger.info("비즈니스 룰 검증")
        
        # 로또 비즈니스 규칙들
        number_cols = [col for col in df.columns if col in ['num1', 'num2', 'num3', 'num4', 'num5', 'num6']]
        
        if len(number_cols) >= 6:
            for idx, row in df.iterrows():
                numbers = [row[col] for col in number_cols[:6] if pd.notna(row[col])]
                
                if len(numbers) == 6:
                    # 연속번호가 3개 이상인지 확인 (매우 드문 경우)
                    sorted_numbers = sorted(numbers)
                    consecutive_count = 0
                    max_consecutive = 0
                    
                    for i in range(len(sorted_numbers) - 1):
                        if sorted_numbers[i+1] - sorted_numbers[i] == 1:
                            consecutive_count += 1
                        else:
                            max_consecutive = max(max_consecutive, consecutive_count + 1)
                            consecutive_count = 0
                    max_consecutive = max(max_consecutive, consecutive_count + 1)
                    
                    if max_consecutive >= 4:
                        result.add_warning(f"연속번호 4개 이상: {sorted_numbers}", idx)
                    
                    # 모든 번호가 홀수이거나 짝수인 경우 (매우 드문 경우)
                    odd_count = sum(1 for n in numbers if n % 2 == 1)
                    if odd_count == 0 or odd_count == 6:
                        result.add_warning(f"모든 번호가 {'홀수' if odd_count == 6 else '짝수'}: {numbers}", idx)
                    
                    # 모든 번호가 특정 구간에 집중된 경우
                    low_count = sum(1 for n in numbers if n <= 15)
                    high_count = sum(1 for n in numbers if n >= 31)
                    
                    if low_count >= 5:
                        result.add_warning(f"번호가 저구간(1-15)에 집중: {numbers}", idx)
                    elif high_count >= 5:
                        result.add_warning(f"번호가 고구간(31-45)에 집중: {numbers}", idx)
        
        # 1등 당첨금 최소값 확인 (보통 10억원 이상)
        if '1st_prize' in df.columns:
            prize_series = pd.to_numeric(df['1st_prize'], errors='coerce')
            valid_prizes = prize_series.dropna()
            
            if len(valid_prizes) > 0:
                min_reasonable_prize = 1_000_000_000  # 10억원
                too_low_prizes = (valid_prizes < min_reasonable_prize).sum()
                
                if too_low_prizes > 0:
                    result.add_warning(f"1등 당첨금이 10억원 미만인 경우: {too_low_prizes}개")
    
    def _validate_statistical_properties(self, df: pd.DataFrame, result: ValidationResult):
        """통계적 속성 검증"""
        self.logger.info("통계적 속성 검증")
        
        number_cols = [col for col in df.columns if col in ['num1', 'num2', 'num3', 'num4', 'num5', 'num6']]
        
        if number_cols:
            # 모든 당첨번호 수집
            all_numbers = []
            for col in number_cols:
                numbers = pd.to_numeric(df[col], errors='coerce').dropna()
                all_numbers.extend(numbers.tolist())
            
            if all_numbers:
                # 번호 출현 빈도 분석
                number_frequency = Counter(all_numbers)
                
                # 이론적 기댓값 (각 번호가 동일 확률로 나올 경우)
                expected_frequency = len(all_numbers) / 45
                
                # 카이제곱 검정을 위한 편차 계산
                significant_deviations = []
                for number in range(1, 46):
                    observed = number_frequency.get(number, 0)
                    deviation = abs(observed - expected_frequency) / expected_frequency
                    
                    if deviation > self.current_thresholds['number_frequency_deviation']:
                        significant_deviations.append((number, observed, deviation))
                
                if significant_deviations:
                    result.add_info(f"유의한 빈도 편차를 보이는 번호: {len(significant_deviations)}개")
                    
                    # 상위 5개만 표시
                    top_deviations = sorted(significant_deviations, key=lambda x: x[2], reverse=True)[:5]
                    for number, observed, deviation in top_deviations:
                        result.add_warning(f"번호 {number}: 출현빈도 {observed}회 (편차: {deviation:.2%})")
                
                # 번호 분포의 균등성 검사
                frequencies = [number_frequency.get(i, 0) for i in range(1, 46)]
                frequency_std = np.std(frequencies)
                frequency_mean = np.mean(frequencies)
                
                if frequency_mean > 0:
                    coefficient_of_variation = frequency_std / frequency_mean
                    result.statistics['frequency_cv'] = coefficient_of_variation
                    
                    # 변동계수가 너무 크면 경고
                    if coefficient_of_variation > 0.5:
                        result.add_warning(f"번호 출현 빈도의 편차가 큼 (CV: {coefficient_of_variation:.3f})")
    
    def _generate_statistics(self, df: pd.DataFrame, result: ValidationResult):
        """검증 통계 생성"""
        result.statistics.update({
            'total_rows': len(df),
            'total_columns': len(df.columns),
            'data_completeness': (df.notna().sum().sum()) / (len(df) * len(df.columns)),
            'duplicate_rows': df.duplicated().sum(),
            'validation_timestamp': datetime.now().isoformat(),
            'tolerance_level': self.tolerance_level
        })
        
        # 당첨번호 통계
        number_cols = [col for col in df.columns if col in ['num1', 'num2', 'num3', 'num4', 'num5', 'num6']]
        if number_cols:
            all_numbers = []
            for col in number_cols:
                numbers = pd.to_numeric(df[col], errors='coerce').dropna()
                all_numbers.extend(numbers.tolist())
            
            if all_numbers:
                result.statistics['number_statistics'] = {
                    'total_numbers': len(all_numbers),
                    'unique_numbers': len(set(all_numbers)),
                    'min_number': min(all_numbers),
                    'max_number': max(all_numbers),
                    'mean_number': np.mean(all_numbers),
                    'most_common': Counter(all_numbers).most_common(5)
                }
    
    def validate_single_row(self, row_data: Dict) -> ValidationResult:
        """
        단일 행 데이터 검증
        
        Args:
            row_data (Dict): 검증할 행 데이터
            
        Returns:
            ValidationResult: 검증 결과
        """
        result = ValidationResult()
        
        # 당첨번호 검증
        numbers = []
        for i in range(1, 7):
            key = f'num{i}'
            if key in row_data and row_data[key] is not None:
                number = row_data[key]
                if not (1 <= number <= 45):
                    result.add_error(f"번호 범위 오류: {key}={number}")
                numbers.append(number)
        
        # 중복 확인
        if len(numbers) != len(set(numbers)):
            result.add_error(f"중복 당첨번호: {numbers}")
        
        # 보너스 번호 검증
        if 'bonus' in row_data and row_data['bonus'] is not None:
            bonus = row_data['bonus']
            if not (1 <= bonus <= 45):
                result.add_error(f"보너스 번호 범위 오류: {bonus}")
            if bonus in numbers:
                result.add_error(f"보너스 번호가 메인 번호와 중복: {bonus}")
        
        return result
    
    def get_validation_summary(self, results: List[ValidationResult]) -> Dict[str, Any]:
        """
        여러 검증 결과의 요약
        
        Args:
            results (List[ValidationResult]): 검증 결과 목록
            
        Returns:
            Dict[str, Any]: 통합 요약 정보
        """
        total_validations = len(results)
        passed_validations = sum(1 for r in results if r.is_valid)
        
        all_errors = []
        all_warnings = []
        
        for result in results:
            all_errors.extend(result.errors)
            all_warnings.extend(result.warnings)
        
        return {
            'summary': {
                'total_validations': total_validations,
                'passed_validations': passed_validations,
                'failed_validations': total_validations - passed_validations,
                'success_rate': passed_validations / total_validations if total_validations > 0 else 0,
                'total_errors': len(all_errors),
                'total_warnings': len(all_warnings)
            },
            'common_issues': {
                'most_common_errors': Counter(all_errors).most_common(5),
                'most_common_warnings': Counter(all_warnings).most_common(5)
            },
            'recommendations': self._get_validation_recommendations(all_errors, all_warnings)
        }
    
    def _get_validation_recommendations(self, errors: List[str], warnings: List[str]) -> List[str]:
        """검증 기반 권장사항 생성"""
        recommendations = []
        
        # 오류 패턴 분석
        error_patterns = Counter([error.split(':')[0] for error in errors])
        warning_patterns = Counter([warning.split(':')[0] for warning in warnings])
        
        if '번호 범위 오류' in str(errors):
            recommendations.append("당첨번호가 1-45 범위를 벗어나는 경우가 있습니다. 데이터 입력 프로세스를 점검하세요.")
        
        if '중복 당첨번호' in str(errors):
            recommendations.append("중복된 당첨번호가 발견되었습니다. 번호 생성 로직을 확인하세요.")
        
        if '날짜' in str(warnings):
            recommendations.append("날짜 관련 문제가 발견되었습니다. 날짜 형식과 간격을 확인하세요.")
        
        if len(warnings) > len(errors) * 2:
            recommendations.append("경고가 많이 발생했습니다. 허용 수준을 'loose'로 변경하거나 데이터 품질을 개선하세요.")
        
        if not recommendations:
            recommendations.append("데이터 품질이 양호합니다. 정기적인 검증을 계속하세요.")
        
        return recommendations


# === 모듈 테스트 ===
if __name__ == "__main__":
    # 테스트용 데이터 생성
    test_data = {
        'round': [1, 2, 3, 4, 5],
        'num1': [1, 2, 3, 4, 5],
        'num2': [7, 8, 9, 10, 11],
        'num3': [14, 15, 16, 17, 18],
        'num4': [21, 22, 23, 24, 25],
        'num5': [28, 29, 30, 31, 32],
        'num6': [35, 36, 37, 38, 39],
        'bonus': [42, 43, 44, 45, 1],
        '1st_prize': [1000000000, 2000000000, 1500000000, 800000000, 1200000000],
        '1st_winners': [1, 2, 0, 1, 3],
        'draw_date': ['2023-01-07', '2023-01-14', '2023-01-21', '2023-01-28', '2023-02-04']
    }
    
    # 문제가 있는 데이터 추가
    problem_data = {
        'round': [6, 7, 8],
        'num1': [50, 2, 3],      # 범위 오류
        'num2': [7, 8, 3],       # 중복 번호 (3번이 num1과 중복)
        'num3': [14, 15, 16],
        'num4': [21, 22, 23],
        'num5': [28, 29, 30],
        'num6': [35, 36, 37],
        'bonus': [35, 43, 44],   # 보너스가 num6과 중복
        '1st_prize': [-1000000, 2000000000, 1500000000],  # 음수
        '1st_winners': [1.5, 2, 0],  # 소수점
        'draw_date': ['2023-02-11', '2025-01-01', '1900-01-01']  # 미래/과거 날짜
    }
    
    # 데이터 결합
    all_data = {}
    for key in test_data.keys():
        all_data[key] = test_data[key] + problem_data.get(key, [])
    
    test_df = pd.DataFrame(all_data)
    
    print("=== 데이터 검증기 테스트 ===")
    print(f"테스트 데이터:\n{test_df.head()}")
    
    try:
        # 다양한 허용 수준으로 테스트
        tolerance_levels = ['strict', 'medium', 'loose']
        
        for level in tolerance_levels:
            print(f"\n🔍 {level.upper()} 모드 검증:")
            validator = LottoDataValidator(tolerance_level=level)
            result = validator.validate_data(test_df)
            
            summary = result.get_summary()
            print(f"  검증 결과: {'통과' if summary['is_valid'] else '실패'}")
            print(f"  오류: {summary['error_count']}개")
            print(f"  경고: {summary['warning_count']}개")
            
            if summary['errors']:
                print("  주요 오류:")
                for error in summary['errors'][:3]:  # 상위 3개만 표시
                    print(f"    - {error}")
            
            if summary['warnings']:
                print("  주요 경고:")
                for warning in summary['warnings'][:3]:  # 상위 3개만 표시
                    print(f"    - {warning}")
        
        print(f"\n📊 통계 정보:")
        stats = result.statistics
        print(f"  총 행 수: {stats.get('total_rows', 0)}")
        print(f"  데이터 완성도: {stats.get('data_completeness', 0):.2%}")
        
        if 'number_statistics' in stats:
            num_stats = stats['number_statistics']
            print(f"  번호 통계:")
            print(f"    - 총 번호 수: {num_stats['total_numbers']}")
            print(f"    - 평균 번호: {num_stats['mean_number']:.1f}")
            print(f"    - 최빈 번호: {num_stats['most_common'][0] if num_stats['most_common'] else 'None'}")
        
        # 단일 행 검증 테스트
        print(f"\n🔎 단일 행 검증 테스트:")
        single_row = {
            'num1': 1, 'num2': 7, 'num3': 14, 'num4': 21, 'num5': 28, 'num6': 35,
            'bonus': 42
        }
        single_result = validator.validate_single_row(single_row)
        print(f"  정상 데이터: {'통과' if single_result.is_valid else '실패'}")
        
        problem_row = {
            'num1': 1, 'num2': 7, 'num3': 14, 'num4': 21, 'num5': 28, 'num6': 1,  # 중복
            'bonus': 50  # 범위 오류
        }
        problem_result = validator.validate_single_row(problem_row)
        print(f"  문제 데이터: {'통과' if problem_result.is_valid else '실패'}")
        if problem_result.errors:
            print(f"    오류: {', '.join(problem_result.errors)}")
        
        print("\n✅ 데이터 검증기 테스트 완료")
        
    except Exception as e:
        print(f"❌ 테스트 중 오류: {e}")
        import traceback
        traceback.print_exc()
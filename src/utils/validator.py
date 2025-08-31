"""
파일명: src/utils/validator.py
목적: 로또 시스템의 데이터 검증 및 유효성 확인
작성자: AI Assistant
작성일: 2025-08-31
버전: 1.0

주요 클래스/함수:
- LottoDataValidator: 데이터 유효성 검증
- PredictionValidator: 예측 결과 검증
- ConfigValidator: 설정 값 검증
- SystemIntegrityValidator: 시스템 무결성 검증
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any, Optional, Set, Union
from collections import Counter, defaultdict
import math
import logging
from datetime import datetime, timedelta
from pathlib import Path
import sys
import re
import json
import warnings

# 상위 디렉토리 imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.utils.config import Config, LottoConstants

warnings.filterwarnings('ignore')

class ValidationResult:
    """검증 결과를 담는 클래스"""
    
    def __init__(self, is_valid: bool = True):
        self.is_valid = is_valid
        self.errors = []
        self.warnings = []
        self.info = []
        self.details = {}
        self.severity_level = 'info'  # 'info', 'warning', 'error', 'critical'
        
    def add_error(self, message: str, detail: Any = None):
        """오류 추가"""
        self.is_valid = False
        self.errors.append(message)
        if detail:
            self.details[f"error_{len(self.errors)}"] = detail
        self.severity_level = 'error'
    
    def add_warning(self, message: str, detail: Any = None):
        """경고 추가"""
        self.warnings.append(message)
        if detail:
            self.details[f"warning_{len(self.warnings)}"] = detail
        if self.severity_level == 'info':
            self.severity_level = 'warning'
    
    def add_info(self, message: str, detail: Any = None):
        """정보 추가"""
        self.info.append(message)
        if detail:
            self.details[f"info_{len(self.info)}"] = detail
    
    def to_dict(self) -> Dict[str, Any]:
        """딕셔너리로 변환"""
        return {
            'is_valid': self.is_valid,
            'severity_level': self.severity_level,
            'errors': self.errors,
            'warnings': self.warnings,
            'info': self.info,
            'details': self.details,
            'summary': self.get_summary()
        }
    
    def get_summary(self) -> str:
        """검증 결과 요약"""
        if not self.is_valid:
            return f"검증 실패: {len(self.errors)}개 오류, {len(self.warnings)}개 경고"
        elif self.warnings:
            return f"검증 통과: {len(self.warnings)}개 경고"
        else:
            return "검증 완료: 문제 없음"


class LottoDataValidator:
    """로또 데이터 유효성 검증기"""
    
    def __init__(self):
        self.logger = self._setup_logger()
        self.required_columns = ['num1', 'num2', 'num3', 'num4', 'num5', 'num6']
        self.optional_columns = ['bonus', 'round', 'draw_date']
        self.number_range = (1, 45)
        
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
    
    def validate_dataframe(self, df: pd.DataFrame, strict_mode: bool = True) -> ValidationResult:
        """
        데이터프레임 종합 검증
        
        Args:
            df (pd.DataFrame): 검증할 데이터프레임
            strict_mode (bool): 엄격 모드 여부
            
        Returns:
            ValidationResult: 검증 결과
        """
        result = ValidationResult()
        
        try:
            # 1. 기본 구조 검증
            self._validate_basic_structure(df, result, strict_mode)
            
            # 2. 컬럼 검증
            self._validate_columns(df, result, strict_mode)
            
            # 3. 데이터 타입 검증
            self._validate_data_types(df, result)
            
            # 4. 번호 범위 검증
            self._validate_number_ranges(df, result)
            
            # 5. 번호 중복 검증
            self._validate_number_duplicates(df, result)
            
            # 6. 시간 데이터 검증
            self._validate_time_data(df, result)
            
            # 7. 데이터 품질 검증
            self._validate_data_quality(df, result)
            
            # 8. 통계적 검증
            self._validate_statistical_properties(df, result)
            
            self.logger.info(f"데이터 검증 완료: {result.get_summary()}")
            
        except Exception as e:
            result.add_error(f"검증 중 예외 발생: {e}")
            self.logger.error(f"데이터 검증 오류: {e}")
        
        return result
    
    def _validate_basic_structure(self, df: pd.DataFrame, result: ValidationResult, strict_mode: bool):
        """기본 구조 검증"""
        # 데이터프레임 존재 확인
        if df is None:
            result.add_error("데이터프레임이 None입니다")
            return
        
        if not isinstance(df, pd.DataFrame):
            result.add_error(f"DataFrame이 아닌 타입입니다: {type(df)}")
            return
        
        # 크기 검증
        if df.empty:
            result.add_error("데이터프레임이 비어있습니다")
            return
        
        if len(df) < 10:
            if strict_mode:
                result.add_error(f"데이터가 너무 적습니다: {len(df)}행 (최소 10행 필요)")
            else:
                result.add_warning(f"데이터가 적습니다: {len(df)}행")
        
        result.add_info(f"데이터 크기: {len(df)}행 × {len(df.columns)}열")
    
    def _validate_columns(self, df: pd.DataFrame, result: ValidationResult, strict_mode: bool):
        """컬럼 검증"""
        existing_columns = set(df.columns)
        required_columns = set(self.required_columns)
        
        # 필수 컬럼 확인
        missing_required = required_columns - existing_columns
        if missing_required:
            result.add_error(f"필수 컬럼 누락: {list(missing_required)}")
        
        # 선택적 컬럼 확인
        missing_optional = set(self.optional_columns) - existing_columns
        if missing_optional:
            result.add_warning(f"선택적 컬럼 누락: {list(missing_optional)}")
        
        # 예상치 못한 컬럼 확인
        expected_columns = set(self.required_columns + self.optional_columns)
        unexpected_columns = existing_columns - expected_columns
        if unexpected_columns:
            result.add_info(f"추가 컬럼 발견: {list(unexpected_columns)}")
        
        result.add_info(f"컬럼 확인: 필수 {len(required_columns - missing_required)}/{len(required_columns)}")
    
    def _validate_data_types(self, df: pd.DataFrame, result: ValidationResult):
        """데이터 타입 검증"""
        # 번호 컬럼들은 정수여야 함
        for col in self.required_columns:
            if col in df.columns:
                # NaN이 아닌 값들의 타입 확인
                non_null_values = df[col].dropna()
                if len(non_null_values) > 0:
                    # 정수로 변환 가능한지 확인
                    try:
                        converted = pd.to_numeric(non_null_values, errors='coerce')
                        invalid_count = converted.isna().sum()
                        
                        if invalid_count > 0:
                            result.add_error(f"{col} 컬럼에 숫자가 아닌 값 {invalid_count}개 발견")
                        
                        # 소수점 값 확인
                        numeric_values = converted.dropna()
                        if len(numeric_values) > 0:
                            decimal_values = numeric_values[numeric_values != numeric_values.astype(int)]
                            if len(decimal_values) > 0:
                                result.add_warning(f"{col} 컬럼에 소수점 값 {len(decimal_values)}개 발견")
                    
                    except Exception as e:
                        result.add_error(f"{col} 컬럼 타입 검증 실패: {e}")
        
        # 보너스 번호 검증
        if 'bonus' in df.columns:
            try:
                bonus_values = pd.to_numeric(df['bonus'], errors='coerce')
                invalid_bonus = bonus_values.isna().sum()
                if invalid_bonus > 0:
                    result.add_warning(f"보너스 번호에 유효하지 않은 값 {invalid_bonus}개")
            except Exception:
                pass
        
        # 날짜 컬럼 검증
        if 'draw_date' in df.columns:
            try:
                pd.to_datetime(df['draw_date'], errors='coerce')
            except Exception as e:
                result.add_warning(f"날짜 형식 검증 실패: {e}")
    
    def _validate_number_ranges(self, df: pd.DataFrame, result: ValidationResult):
        """번호 범위 검증"""
        min_num, max_num = self.number_range
        
        for col in self.required_columns:
            if col in df.columns:
                # 숫자로 변환 후 범위 확인
                try:
                    numbers = pd.to_numeric(df[col], errors='coerce').dropna()
                    
                    # 범위 밖 값 확인
                    out_of_range = numbers[(numbers < min_num) | (numbers > max_num)]
                    
                    if len(out_of_range) > 0:
                        result.add_error(f"{col} 컬럼에 범위 밖 값 {len(out_of_range)}개: "
                                       f"범위 {min_num}-{max_num} 벗어남")
                        result.details[f"{col}_out_of_range"] = out_of_range.tolist()[:10]  # 최대 10개만
                    
                    # 범위 분포 정보
                    if len(numbers) > 0:
                        result.add_info(f"{col} 범위: {numbers.min():.0f} ~ {numbers.max():.0f}")
                
                except Exception as e:
                    result.add_error(f"{col} 범위 검증 실패: {e}")
        
        # 보너스 번호 범위 검증
        if 'bonus' in df.columns:
            try:
                bonus_numbers = pd.to_numeric(df['bonus'], errors='coerce').dropna()
                out_of_range_bonus = bonus_numbers[(bonus_numbers < min_num) | (bonus_numbers > max_num)]
                
                if len(out_of_range_bonus) > 0:
                    result.add_error(f"보너스 번호에 범위 밖 값 {len(out_of_range_bonus)}개")
            
            except Exception:
                pass
    
    def _validate_number_duplicates(self, df: pd.DataFrame, result: ValidationResult):
        """번호 중복 검증"""
        duplicate_rows = 0
        duplicate_examples = []
        
        for idx, row in df.iterrows():
            # 당첨번호 6개 추출
            numbers = []
            for col in self.required_columns:
                if col in df.columns and pd.notna(row[col]):
                    try:
                        num = int(float(row[col]))
                        numbers.append(num)
                    except (ValueError, TypeError):
                        continue
            
            # 6개 번호가 모두 있는 경우만 검증
            if len(numbers) == 6:
                # 중복 번호 확인
                if len(set(numbers)) != len(numbers):
                    duplicate_rows += 1
                    if len(duplicate_examples) < 5:
                        duplicates = [num for num in numbers if numbers.count(num) > 1]
                        duplicate_examples.append((idx, numbers, list(set(duplicates))))
                
                # 정렬 확인 (로또는 일반적으로 오름차순)
                if numbers != sorted(numbers):
                    result.add_warning(f"행 {idx}: 번호가 정렬되지 않음 {numbers}")
        
        if duplicate_rows > 0:
            result.add_error(f"중복 번호가 있는 행: {duplicate_rows}개")
            result.details['duplicate_examples'] = duplicate_examples
        
        result.add_info(f"중복 검증: {len(df) - duplicate_rows}/{len(df)} 행 정상")
    
    def _validate_time_data(self, df: pd.DataFrame, result: ValidationResult):
        """시간 데이터 검증"""
        if 'draw_date' in df.columns:
            try:
                # 날짜 파싱 시도
                dates = pd.to_datetime(df['draw_date'], errors='coerce')
                invalid_dates = dates.isna().sum()
                
                if invalid_dates > 0:
                    result.add_warning(f"유효하지 않은 날짜 {invalid_dates}개")
                
                # 날짜 범위 확인
                valid_dates = dates.dropna()
                if len(valid_dates) > 0:
                    date_range = (valid_dates.min(), valid_dates.max())
                    result.add_info(f"날짜 범위: {date_range[0].strftime('%Y-%m-%d')} ~ {date_range[1].strftime('%Y-%m-%d')}")
                    
                    # 날짜 순서 확인
                    if not valid_dates.is_monotonic_increasing:
                        result.add_warning("날짜가 시간 순으로 정렬되지 않음")
                    
                    # 날짜 간격 확인
                    date_gaps = valid_dates.diff().dropna()
                    if len(date_gaps) > 0:
                        avg_gap = date_gaps.mean()
                        std_gap = date_gaps.std()
                        
                        # 일반적인 로또 추첨은 주 2회 (3-4일 간격)
                        expected_gap = timedelta(days=3.5)
                        gap_deviation = abs(avg_gap - expected_gap)
                        
                        if gap_deviation > timedelta(days=2):
                            result.add_warning(f"비정상적인 날짜 간격: 평균 {avg_gap}")
            
            except Exception as e:
                result.add_warning(f"날짜 검증 실패: {e}")
        
        # 회차 번호 검증
        if 'round' in df.columns:
            try:
                rounds = pd.to_numeric(df['round'], errors='coerce').dropna()
                
                if len(rounds) > 0:
                    # 회차 순서 확인
                    if not rounds.is_monotonic_increasing:
                        result.add_warning("회차 번호가 순서대로 정렬되지 않음")
                    
                    # 회차 연속성 확인
                    round_gaps = rounds.diff().dropna()
                    non_unit_gaps = round_gaps[round_gaps != 1]
                    
                    if len(non_unit_gaps) > 0:
                        result.add_warning(f"회차 번호 불연속 구간: {len(non_unit_gaps)}개")
                    
                    result.add_info(f"회차 범위: {int(rounds.min())} ~ {int(rounds.max())}")
            
            except Exception as e:
                result.add_warning(f"회차 검증 실패: {e}")
    
    def _validate_data_quality(self, df: pd.DataFrame, result: ValidationResult):
        """데이터 품질 검증"""
        # 결측값 확인
        missing_data = df.isnull().sum()
        
        for col in self.required_columns:
            if col in missing_data and missing_data[col] > 0:
                missing_ratio = missing_data[col] / len(df)
                if missing_ratio > 0.05:  # 5% 이상 결측
                    result.add_error(f"{col} 컬럼 결측값 과다: {missing_ratio:.1%}")
                else:
                    result.add_warning(f"{col} 컬럼 결측값: {missing_data[col]}개")
        
        # 데이터 일관성 확인
        self._validate_data_consistency(df, result)
        
        # 이상치 확인
        self._validate_outliers(df, result)
    
    def _validate_data_consistency(self, df: pd.DataFrame, result: ValidationResult):
        """데이터 일관성 검증"""
        inconsistent_rows = []
        
        for idx, row in df.iterrows():
            numbers = []
            for col in self.required_columns:
                if col in df.columns and pd.notna(row[col]):
                    try:
                        num = int(float(row[col]))
                        numbers.append(num)
                    except (ValueError, TypeError):
                        inconsistent_rows.append((idx, f"{col} 값이 숫자가 아님: {row[col]}"))
                        continue
            
            # 6개 번호가 모두 있어야 함
            if len(numbers) != 6:
                inconsistent_rows.append((idx, f"번호 개수 부족: {len(numbers)}개"))
            
            # 보너스 번호와 당첨번호 중복 확인
            if 'bonus' in df.columns and pd.notna(row['bonus']):
                try:
                    bonus_num = int(float(row['bonus']))
                    if bonus_num in numbers:
                        inconsistent_rows.append((idx, f"보너스 번호가 당첨번호와 중복: {bonus_num}"))
                except (ValueError, TypeError):
                    inconsistent_rows.append((idx, f"보너스 번호가 숫자가 아님: {row['bonus']}"))
        
        if inconsistent_rows:
            result.add_error(f"일관성 오류: {len(inconsistent_rows)}행")
            result.details['consistency_errors'] = inconsistent_rows[:10]  # 최대 10개
    
    def _validate_outliers(self, df: pd.DataFrame, result: ValidationResult):
        """이상치 검증"""
        outlier_analysis = {}
        
        # 각 번호 위치별 이상치 확인
        for col in self.required_columns:
            if col in df.columns:
                try:
                    numbers = pd.to_numeric(df[col], errors='coerce').dropna()
                    
                    if len(numbers) > 10:
                        # IQR 방법으로 이상치 탐지
                        Q1 = numbers.quantile(0.25)
                        Q3 = numbers.quantile(0.75)
                        IQR = Q3 - Q1
                        
                        lower_bound = Q1 - 1.5 * IQR
                        upper_bound = Q3 + 1.5 * IQR
                        
                        outliers = numbers[(numbers < lower_bound) | (numbers > upper_bound)]
                        
                        if len(outliers) > 0:
                            outlier_ratio = len(outliers) / len(numbers)
                            if outlier_ratio > 0.1:  # 10% 이상이 이상치
                                result.add_warning(f"{col} 컬럼 이상치 과다: {outlier_ratio:.1%}")
                            
                            outlier_analysis[col] = {
                                'count': len(outliers),
                                'ratio': outlier_ratio,
                                'examples': outliers.tolist()[:5]
                            }
                
                except Exception:
                    pass
        
        if outlier_analysis:
            result.details['outlier_analysis'] = outlier_analysis
    
    def _validate_statistical_properties(self, df: pd.DataFrame, result: ValidationResult):
        """통계적 속성 검증"""
        try:
            # 모든 당첨번호 수집
            all_numbers = []
            for col in self.required_columns:
                if col in df.columns:
                    numbers = pd.to_numeric(df[col], errors='coerce').dropna()
                    all_numbers.extend(numbers.tolist())
        
            if len(all_numbers) >= 30:  # 최소 샘플 크기
                # 기본 통계
                stats_info = {
                    'count': len(all_numbers),
                    'mean': np.mean(all_numbers),
                    'std': np.std(all_numbers),
                    'min': np.min(all_numbers),
                    'max': np.max(all_numbers)
                }
            
                result.add_info(f"통계 요약: 평균 {stats_info['mean']:.1f}, 표준편차 {stats_info['std']:.1f}")
            
                # 분포 균등성 검증
                number_counts = Counter(all_numbers)
                expected_count = len(all_numbers) / 45
            
                # 카이제곱 검정
                observed_counts = [number_counts.get(i, 0) for i in range(1, 46)]
                expected_counts = [expected_count] * 45
            
                try:
                    from scipy.stats import chisquare
                    chi2_stat, p_value = chisquare(observed_counts, expected_counts)
                
                    if p_value < 0.01:  # 매우 유의한 차이
                        result.add_warning(f"번호 분포가 균등하지 않음 (p={p_value:.4f})")
                
                    result.details['distribution_test'] = {
                        'chi2_statistic': chi2_stat,
                        'p_value': p_value,
                        'uniform_hypothesis': 'rejected' if p_value < 0.05 else 'not_rejected'
                    }
            
                except ImportError:
                    # scipy 없으면 단순 방법
                    max_count = max(observed_counts)
                    min_count = min(observed_counts)
                    ratio = max_count / min_count if min_count > 0 else float('inf')
                
                    if ratio > 3:  # 3배 이상 차이
                        result.add_warning(f"번호별 출현 빈도 차이 큼: 최대/최소 = {ratio:.1f}")
            
                except Exception as e:
                    result.add_warning(f"분포 검정 실패: {e}")
        
            else:
                result.add_warning("통계적 검증을 위한 데이터 부족")
    
        except Exception as e:
            result.add_warning(f"통계적 속성 검증 실패: {e}")


class PredictionValidator:
    """예측 결과 검증기"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.number_range = (1, 45)
        self.required_prediction_count = 6
    
    def validate_prediction(self, numbers: Union[List[int], np.ndarray], 
                          context: Dict = None) -> ValidationResult:
        """
        예측 번호 검증
        
        Args:
            numbers: 예측된 번호들
            context: 추가 컨텍스트 정보
            
        Returns:
            ValidationResult: 검증 결과
        """
        result = ValidationResult()
        
        try:
            # 기본 타입 검증
            self._validate_prediction_format(numbers, result)
            
            if result.is_valid:
                # 번호 유효성 검증
                self._validate_prediction_numbers(numbers, result)
                
                # 예측 품질 검증
                self._validate_prediction_quality(numbers, result, context)
                
                # 예측 다양성 검증
                self._validate_prediction_diversity(numbers, result, context)
        
        except Exception as e:
            result.add_error(f"예측 검증 중 오류: {e}")
        
        return result
    
    def _validate_prediction_format(self, numbers: Any, result: ValidationResult):
        """예측 형식 검증"""
        if numbers is None:
            result.add_error("예측 결과가 None입니다")
            return
        
        # 리스트나 배열로 변환
        try:
            if isinstance(numbers, np.ndarray):
                number_list = numbers.tolist()
            elif isinstance(numbers, (list, tuple)):
                number_list = list(numbers)
            else:
                result.add_error(f"예측 형식이 잘못됨: {type(numbers)}")
                return
            
            # 개수 확인
            if len(number_list) != self.required_prediction_count:
                result.add_error(f"예측 번호 개수 오류: {len(number_list)}개 (6개 필요)")
                return
            
            # 숫자 타입 확인
            for i, num in enumerate(number_list):
                try:
                    int_num = int(float(num))
                    number_list[i] = int_num
                except (ValueError, TypeError):
                    result.add_error(f"번호 {i+1}이 숫자가 아님: {num}")
                    return
            
            result.details['validated_numbers'] = number_list
            
        except Exception as e:
            result.add_error(f"형식 검증 실패: {e}")
    
    def _validate_prediction_numbers(self, numbers: List[int], result: ValidationResult):
        """예측 번호 유효성 검증"""
        validated_numbers = result.details.get('validated_numbers', numbers)
        
        # 범위 검증
        min_num, max_num = self.number_range
        for i, num in enumerate(validated_numbers):
            if not (min_num <= num <= max_num):
                result.add_error(f"번호 {i+1}이 범위 밖: {num} (범위: {min_num}-{max_num})")
        
        # 중복 검증
        if len(set(validated_numbers)) != len(validated_numbers):
            duplicates = [num for num in validated_numbers if validated_numbers.count(num) > 1]
            result.add_error(f"중복 번호 발견: {list(set(duplicates))}")
        
        # 정렬 상태 확인
        if validated_numbers != sorted(validated_numbers):
            result.add_info("번호가 정렬되지 않음 (자동 정렬 권장)")
            result.details['sorted_numbers'] = sorted(validated_numbers)
    
    def _validate_prediction_quality(self, numbers: List[int], result: ValidationResult, context: Dict = None):
        """예측 품질 검증"""
        validated_numbers = result.details.get('validated_numbers', numbers)
        
        if not result.is_valid:
            return
        
        # 기본 품질 지표 계산
        quality_metrics = self._calculate_quality_metrics(validated_numbers)
        result.details['quality_metrics'] = quality_metrics
        
        # 품질 기준 검증
        self._check_quality_standards(quality_metrics, result)
        
        # 컨텍스트 기반 검증
        if context:
            self._validate_with_context(validated_numbers, result, context)
    
    def _calculate_quality_metrics(self, numbers: List[int]) -> Dict:
        """예측 품질 지표 계산"""
        sorted_numbers = sorted(numbers)
        
        metrics = {
            'sum': sum(numbers),
            'mean': np.mean(numbers),
            'std': np.std(numbers),
            'range': max(numbers) - min(numbers),
            'odd_count': sum(1 for n in numbers if n % 2 == 1),
            'even_count': sum(1 for n in numbers if n % 2 == 0),
            'consecutive_pairs': self._count_consecutive_pairs(sorted_numbers),
            'section_distribution': self._analyze_section_distribution(numbers)
        }
        
        # 추가 지표
        metrics['odd_ratio'] = metrics['odd_count'] / len(numbers)
        metrics['balance_score'] = self._calculate_balance_score(numbers)
        
        return metrics
    
    def _count_consecutive_pairs(self, sorted_numbers: List[int]) -> int:
        """연속 번호 쌍 개수 계산"""
        consecutive_pairs = 0
        for i in range(len(sorted_numbers) - 1):
            if sorted_numbers[i+1] - sorted_numbers[i] == 1:
                consecutive_pairs += 1
        return consecutive_pairs
    
    def _analyze_section_distribution(self, numbers: List[int]) -> Dict:
        """구간별 분포 분석"""
        sections = {
            'low': (1, 15),
            'mid': (16, 30),
            'high': (31, 45)
        }
        
        distribution = {}
        for section_name, (start, end) in sections.items():
            count = sum(1 for num in numbers if start <= num <= end)
            distribution[section_name] = count
        
        return distribution
    
    def _calculate_balance_score(self, numbers: List[int]) -> float:
        """번호 균형 점수 계산 (0-1, 1이 가장 균형)"""
        # 구간별 분포 균형
        section_dist = self._analyze_section_distribution(numbers)
        ideal_per_section = len(numbers) / 3  # 2개씩
        
        section_variance = np.var(list(section_dist.values()))
        balance_score = 1 / (1 + section_variance)  # 분산이 작을수록 균형
        
        return balance_score
    
    def _check_quality_standards(self, metrics: Dict, result: ValidationResult):
        """품질 기준 검증"""
        # 합계 범위 검증 (일반적으로 80-200 사이)
        if not (80 <= metrics['sum'] <= 200):
            result.add_warning(f"번호 합계가 비정상적: {metrics['sum']} (일반 범위: 80-200)")
        
        # 홀짝 비율 검증 (너무 편중되지 않아야 함)
        if metrics['odd_ratio'] < 0.17 or metrics['odd_ratio'] > 0.83:  # 1개 또는 5개 이상
            result.add_warning(f"홀짝 비율이 극단적: 홀수 {metrics['odd_count']}개")
        
        # 연속 번호 과다 검증
        if metrics['consecutive_pairs'] >= 3:
            result.add_warning(f"연속 번호 쌍이 과다: {metrics['consecutive_pairs']}쌍")
        
        # 구간 분포 검증
        section_dist = metrics['section_distribution']
        max_section = max(section_dist.values())
        if max_section >= 5:  # 한 구간에 5개 이상
            result.add_warning(f"특정 구간 집중도 과다: {max_section}개")
        
        # 균형 점수 검증
        if metrics['balance_score'] < 0.3:
            result.add_warning(f"번호 균형도 낮음: {metrics['balance_score']:.3f}")
    
    def _validate_with_context(self, numbers: List[int], result: ValidationResult, context: Dict):
        """컨텍스트 기반 검증"""
        # 과거 예측과의 중복 확인
        if 'recent_predictions' in context:
            recent_predictions = context['recent_predictions']
            for i, past_prediction in enumerate(recent_predictions[-5:]):  # 최근 5개
                if sorted(numbers) == sorted(past_prediction):
                    result.add_error(f"최근 {5-i}번째 예측과 완전 동일")
                    break
        
        # 예측 신뢰도 확인
        if 'confidence' in context:
            confidence = context['confidence']
            if confidence < 0.1:
                result.add_warning(f"예측 신뢰도가 매우 낮음: {confidence:.3f}")
            elif confidence > 0.9:
                result.add_warning(f"예측 신뢰도가 비현실적으로 높음: {confidence:.3f}")
        
        # 사용된 공식 정보 확인
        if 'formulas_used' in context:
            formulas = context['formulas_used']
            if len(formulas) < 2:
                result.add_warning("사용된 공식이 너무 적음 (다양성 부족 가능)")
    
    def _validate_prediction_diversity(self, numbers: List[int], result: ValidationResult, context: Dict = None):
        """예측 다양성 검증"""
        # 번호 간 간격 분석
        sorted_numbers = sorted(numbers)
        gaps = [sorted_numbers[i+1] - sorted_numbers[i] for i in range(5)]
        
        # 간격이 너무 균등하거나 불균등한지 확인
        gap_std = np.std(gaps)
        if gap_std < 2:
            result.add_warning(f"번호 간격이 너무 균등함: 표준편차 {gap_std:.1f}")
        elif gap_std > 15:
            result.add_warning(f"번호 간격이 너무 불균등함: 표준편차 {gap_std:.1f}")
        
        result.details['gap_analysis'] = {
            'gaps': gaps,
            'gap_std': gap_std,
            'gap_mean': np.mean(gaps)
        }


class ConfigValidator:
    """설정 값 검증기"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def validate_config(self, config_data: Dict) -> ValidationResult:
        """
        설정 데이터 검증
        
        Args:
            config_data: 검증할 설정 데이터
            
        Returns:
            ValidationResult: 검증 결과
        """
        result = ValidationResult()
        
        try:
            # 필수 설정 확인
            self._validate_required_settings(config_data, result)
            
            # 값 범위 검증
            self._validate_value_ranges(config_data, result)
            
            # 설정 일관성 검증
            self._validate_config_consistency(config_data, result)
            
        except Exception as e:
            result.add_error(f"설정 검증 중 오류: {e}")
        
        return result
    
    def _validate_required_settings(self, config: Dict, result: ValidationResult):
        """필수 설정 확인"""
        required_keys = [
            'data_file_path',
            'prediction_count',
            'confidence_threshold',
            'formula_weights'
        ]
        
        for key in required_keys:
            if key not in config:
                result.add_error(f"필수 설정 누락: {key}")
            elif config[key] is None:
                result.add_error(f"필수 설정이 None: {key}")
    
    def _validate_value_ranges(self, config: Dict, result: ValidationResult):
        """값 범위 검증"""
        range_rules = {
            'prediction_count': (1, 10),
            'confidence_threshold': (0.0, 1.0),
            'min_confidence': (0.0, 1.0),
            'max_iterations': (1, 10000),
            'forecast_horizon': (1, 100)
        }
        
        for key, (min_val, max_val) in range_rules.items():
            if key in config:
                value = config[key]
                try:
                    numeric_value = float(value)
                    if not (min_val <= numeric_value <= max_val):
                        result.add_error(f"{key} 값이 범위 밖: {value} (범위: {min_val}-{max_val})")
                except (ValueError, TypeError):
                    result.add_error(f"{key} 값이 숫자가 아님: {value}")
    
    def _validate_config_consistency(self, config: Dict, result: ValidationResult):
        """설정 일관성 검증"""
        # 가중치 검증
        if 'formula_weights' in config:
            weights = config['formula_weights']
            if isinstance(weights, dict):
                weight_sum = sum(weights.values())
                if abs(weight_sum - 1.0) > 0.01:  # 1% 허용 오차
                    result.add_warning(f"공식 가중치 합이 1이 아님: {weight_sum:.3f}")
                
                # 음수 가중치 확인
                negative_weights = {k: v for k, v in weights.items() if v < 0}
                if negative_weights:
                    result.add_warning(f"음수 가중치 발견: {negative_weights}")
        
        # 파일 경로 검증
        if 'data_file_path' in config:
            file_path = Path(config['data_file_path'])
            if not file_path.exists():
                result.add_error(f"데이터 파일이 존재하지 않음: {file_path}")
            elif not file_path.is_file():
                result.add_error(f"데이터 경로가 파일이 아님: {file_path}")


class SystemIntegrityValidator:
    """시스템 무결성 검증기"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def validate_system_integrity(self, components: Dict) -> ValidationResult:
        """
        시스템 전체 무결성 검증
        
        Args:
            components: 시스템 컴포넌트들
            
        Returns:
            ValidationResult: 검증 결과
        """
        result = ValidationResult()
        
        try:
            # 컴포넌트 존재 확인
            self._validate_component_existence(components, result)
            
            # 컴포넌트 간 연결성 확인
            self._validate_component_connections(components, result)
            
            # 데이터 플로우 검증
            self._validate_data_flow(components, result)
            
            # 성능 검증
            self._validate_performance(components, result)
            
        except Exception as e:
            result.add_error(f"시스템 무결성 검증 중 오류: {e}")
        
        return result
    
    def _validate_component_existence(self, components: Dict, result: ValidationResult):
        """컴포넌트 존재 확인"""
        required_components = [
            'data_loader',
            'statistics_analyzer', 
            'pattern_detector',
            'formula_engine',
            'predictor'
        ]
        
        for component_name in required_components:
            if component_name not in components:
                result.add_error(f"필수 컴포넌트 누락: {component_name}")
            elif components[component_name] is None:
                result.add_error(f"컴포넌트가 초기화되지 않음: {component_name}")
    
    def _validate_component_connections(self, components: Dict, result: ValidationResult):
        """컴포넌트 간 연결성 확인"""
        # 데이터 의존성 확인
        dependencies = {
            'statistics_analyzer': ['data_loader'],
            'pattern_detector': ['data_loader', 'statistics_analyzer'],
            'formula_engine': ['statistics_analyzer', 'pattern_detector'],
            'predictor': ['formula_engine']
        }
        
        for component, deps in dependencies.items():
            if component in components:
                for dep in deps:
                    if dep not in components or components[dep] is None:
                        result.add_error(f"{component}의 의존성 누락: {dep}")
    
    def _validate_data_flow(self, components: Dict, result: ValidationResult):
        """데이터 플로우 검증"""
        try:
            # 샘플 데이터로 플로우 테스트
            if 'data_loader' in components and components['data_loader']:
                # 데이터 로딩 테스트
                try:
                    sample_data = components['data_loader'].load_sample_data()
                    if sample_data is None or sample_data.empty:
                        result.add_error("데이터 로딩 실패")
                    else:
                        result.add_info(f"데이터 플로우 테스트: {len(sample_data)}행 로딩 성공")
                except Exception as e:
                    result.add_error(f"데이터 로딩 테스트 실패: {e}")
        
        except Exception as e:
            result.add_warning(f"데이터 플로우 검증 실패: {e}")
    
    def _validate_performance(self, components: Dict, result: ValidationResult):
        """성능 검증"""
        performance_issues = []
        
        # 메모리 사용량 체크 (간단한 추정)
        try:
            import sys
            total_size = 0
            for component_name, component in components.items():
                if component:
                    component_size = sys.getsizeof(component)
                    total_size += component_size
            
            # 100MB 이상이면 경고
            if total_size > 100 * 1024 * 1024:
                result.add_warning(f"시스템 메모리 사용량 높음: {total_size / (1024*1024):.1f}MB")
            
            result.add_info(f"추정 메모리 사용량: {total_size / (1024*1024):.1f}MB")
        
        except Exception:
            pass


class PipelineValidator:
    """전체 파이프라인 검증기"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.data_validator = LottoDataValidator()
        self.prediction_validator = PredictionValidator()
        self.config_validator = ConfigValidator()
        self.system_validator = SystemIntegrityValidator()
    
    def validate_complete_pipeline(self, data: pd.DataFrame, config: Dict, 
                                 components: Dict, test_prediction: bool = True) -> Dict[str, ValidationResult]:
        """
        전체 파이프라인 검증
        
        Args:
            data: 로또 데이터
            config: 설정 데이터
            components: 시스템 컴포넌트들
            test_prediction: 예측 테스트 수행 여부
            
        Returns:
            Dict[str, ValidationResult]: 분야별 검증 결과
        """
        self.logger.info("전체 파이프라인 검증 시작")
        
        results = {}
        
        try:
            # 1. 데이터 검증
            self.logger.info("데이터 검증")
            results['data'] = self.data_validator.validate_dataframe(data)
            
            # 2. 설정 검증
            self.logger.info("설정 검증")
            results['config'] = self.config_validator.validate_config(config)
            
            # 3. 시스템 무결성 검증
            self.logger.info("시스템 무결성 검증")
            results['system'] = self.system_validator.validate_system_integrity(components)
            
            # 4. 예측 테스트 (선택적)
            if test_prediction and all(r.is_valid for r in results.values()):
                self.logger.info("예측 시스템 테스트")
                results['prediction'] = self._test_prediction_system(components, data)
            
            # 5. 통합 검증 결과
            self.logger.info("통합 검증")
            results['integration'] = self._validate_integration(results)
            
            self.logger.info("전체 파이프라인 검증 완료")
            
        except Exception as e:
            error_result = ValidationResult()
            error_result.add_error(f"파이프라인 검증 중 오류: {e}")
            results['pipeline_error'] = error_result
            self.logger.error(f"파이프라인 검증 오류: {e}")
        
        return results
    
    def _test_prediction_system(self, components: Dict, data: pd.DataFrame) -> ValidationResult:
        """예측 시스템 테스트"""
        result = ValidationResult()
        
        try:
            # 예측기가 있는지 확인
            if 'predictor' not in components or not components['predictor']:
                result.add_error("예측기 컴포넌트가 없음")
                return result
            
            predictor = components['predictor']
            
            # 테스트 예측 수행
            test_predictions = []
            for i in range(3):  # 3회 테스트
                try:
                    prediction = predictor.predict_numbers(6)
                    test_predictions.append(prediction)
                    
                    # 개별 예측 검증
                    pred_validation = self.prediction_validator.validate_prediction(prediction)
                    if not pred_validation.is_valid:
                        result.add_error(f"테스트 예측 {i+1} 실패: {pred_validation.errors}")
                
                except Exception as e:
                    result.add_error(f"테스트 예측 {i+1} 생성 실패: {e}")
            
            # 예측 다양성 확인
            if len(test_predictions) >= 2:
                identical_predictions = 0
                for i in range(len(test_predictions)):
                    for j in range(i+1, len(test_predictions)):
                        if sorted(test_predictions[i]) == sorted(test_predictions[j]):
                            identical_predictions += 1
                
                if identical_predictions > 0:
                    result.add_error(f"동일한 테스트 예측 {identical_predictions}쌍 발견")
                else:
                    result.add_info("예측 다양성 테스트 통과")
            
            result.add_info(f"예측 시스템 테스트: {len(test_predictions)}회 수행")
        
        except Exception as e:
            result.add_error(f"예측 시스템 테스트 실패: {e}")
        
        return result
    
    def _validate_integration(self, results: Dict[str, ValidationResult]) -> ValidationResult:
        """통합 검증"""
        integration_result = ValidationResult()
        
        # 전체 오류 수집
        total_errors = 0
        total_warnings = 0
        critical_issues = []
        
        for validation_type, validation_result in results.items():
            if validation_type == 'integration':  # 자기 자신 제외
                continue
                
            total_errors += len(validation_result.errors)
            total_warnings += len(validation_result.warnings)
            
            # 중요한 오류 식별
            for error in validation_result.errors:
                if any(keyword in error.lower() for keyword in ['누락', '실패', '중복', '범위']):
                    critical_issues.append(f"{validation_type}: {error}")
        
        # 통합 평가
        if total_errors == 0:
            if total_warnings == 0:
                integration_result.add_info("시스템이 완벽하게 검증됨")
            else:
                integration_result.add_warning(f"경미한 문제 {total_warnings}개 있으나 동작 가능")
        else:
            integration_result.add_error(f"시스템에 {total_errors}개 오류 있음")
            
            if critical_issues:
                integration_result.add_error("중요한 문제들:")
                for issue in critical_issues[:5]:  # 최대 5개
                    integration_result.add_error(f"  - {issue}")
        
        # 시스템 상태 평가
        if total_errors == 0 and total_warnings <= 3:
            system_status = "excellent"
        elif total_errors == 0 and total_warnings <= 10:
            system_status = "good"
        elif total_errors <= 2:
            system_status = "needs_attention"
        else:
            system_status = "critical"
        
        integration_result.details['system_status'] = system_status
        integration_result.details['total_errors'] = total_errors
        integration_result.details['total_warnings'] = total_warnings
        integration_result.details['critical_issues_count'] = len(critical_issues)
        
        return integration_result


class DataQualityValidator:
    """데이터 품질 전문 검증기"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def validate_data_quality(self, df: pd.DataFrame) -> ValidationResult:
        """데이터 품질 종합 검증"""
        result = ValidationResult()
        
        try:
            # 1. 완전성 검증
            self._validate_completeness(df, result)
            
            # 2. 정확성 검증
            self._validate_accuracy(df, result)
            
            # 3. 일관성 검증
            self._validate_consistency(df, result)
            
            # 4. 유일성 검증
            self._validate_uniqueness(df, result)
            
            # 5. 적시성 검증
            self._validate_timeliness(df, result)
            
            # 6. 타당성 검증
            self._validate_validity(df, result)
            
        except Exception as e:
            result.add_error(f"데이터 품질 검증 중 오류: {e}")
        
        return result
    
    def _validate_completeness(self, df: pd.DataFrame, result: ValidationResult):
        """완전성 검증"""
        required_cols = ['num1', 'num2', 'num3', 'num4', 'num5', 'num6']
        
        for col in required_cols:
            if col in df.columns:
                missing_count = df[col].isnull().sum()
                missing_ratio = missing_count / len(df)
                
                if missing_ratio > 0.01:  # 1% 이상 결측
                    if missing_ratio > 0.05:
                        result.add_error(f"{col} 완전성 부족: {missing_ratio:.1%} 결측")
                    else:
                        result.add_warning(f"{col} 결측값: {missing_ratio:.1%}")
        
        # 전체 완전성 점수
        total_cells = len(df) * len(required_cols)
        complete_cells = total_cells - df[required_cols].isnull().sum().sum()
        completeness_score = complete_cells / total_cells
        
        result.details['completeness_score'] = completeness_score
        result.add_info(f"데이터 완전성: {completeness_score:.1%}")
    
    def _validate_accuracy(self, df: pd.DataFrame, result: ValidationResult):
        """정확성 검증"""
        accuracy_issues = 0
        
        # 로또 규칙 준수 확인
        for idx, row in df.iterrows():
            numbers = []
            for col in ['num1', 'num2', 'num3', 'num4', 'num5', 'num6']:
                if col in df.columns and pd.notna(row[col]):
                    try:
                        num = int(float(row[col]))
                        if 1 <= num <= 45:
                            numbers.append(num)
                        else:
                            accuracy_issues += 1
                    except (ValueError, TypeError):
                        accuracy_issues += 1
            
            # 정확한 6개 번호인지 확인
            if len(numbers) != 6:
                accuracy_issues += 1
            
            # 중복 없는지 확인
            if len(set(numbers)) != len(numbers):
                accuracy_issues += 1
        
        accuracy_ratio = 1 - (accuracy_issues / (len(df) * 6))
        result.details['accuracy_score'] = accuracy_ratio
        
        if accuracy_ratio < 0.95:
            result.add_error(f"데이터 정확성 부족: {accuracy_ratio:.1%}")
        elif accuracy_ratio < 0.99:
            result.add_warning(f"데이터 정확성: {accuracy_ratio:.1%}")
        else:
            result.add_info(f"데이터 정확성: {accuracy_ratio:.1%}")
    
    def _validate_consistency(self, df: pd.DataFrame, result: ValidationResult):
        """일관성 검증"""
        # 동일 회차 중복 확인
        if 'round' in df.columns:
            duplicate_rounds = df['round'].duplicated().sum()
            if duplicate_rounds > 0:
                result.add_error(f"중복 회차: {duplicate_rounds}개")
        
        # 날짜-회차 일관성 확인
        if 'draw_date' in df.columns and 'round' in df.columns:
            try:
                df_temp = df.copy()
                df_temp['draw_date'] = pd.to_datetime(df_temp['draw_date'])
                df_temp = df_temp.dropna(subset=['draw_date', 'round'])
                
                if len(df_temp) > 1:
                    # 날짜 순서와 회차 순서 일치 확인
                    date_sorted = df_temp.sort_values('draw_date')
                    round_sorted = df_temp.sort_values('round')
                    
                    if not date_sorted['round'].equals(round_sorted['round']):
                        result.add_warning("날짜 순서와 회차 순서가 불일치")
            
            except Exception:
                result.add_warning("날짜-회차 일관성 검증 실패")
    
    def _validate_uniqueness(self, df: pd.DataFrame, result: ValidationResult):
        """유일성 검증"""
        # 동일한 번호 조합 확인
        number_cols = ['num1', 'num2', 'num3', 'num4', 'num5', 'num6']
        
        if all(col in df.columns for col in number_cols):
            # 각 행의 번호 조합을 튜플로 변환
            number_combinations = []
            for idx, row in df.iterrows():
                numbers = []
                for col in number_cols:
                    if pd.notna(row[col]):
                        try:
                            numbers.append(int(float(row[col])))
                        except (ValueError, TypeError):
                            numbers.append(-1)  # 오류 표시
                
                if len(numbers) == 6:
                    number_combinations.append(tuple(sorted(numbers)))
            
            # 중복 조합 찾기
            combination_counts = Counter(number_combinations)
            duplicates = [(combo, count) for combo, count in combination_counts.items() if count > 1]
            
            if duplicates:
                result.add_error(f"동일한 번호 조합: {len(duplicates)}개")
                result.details['duplicate_combinations'] = duplicates[:5]
            else:
                result.add_info("모든 번호 조합이 유일함")
    
    def _validate_timeliness(self, df: pd.DataFrame, result: ValidationResult):
        """적시성 검증"""
        if 'draw_date' not in df.columns:
            result.add_warning("날짜 정보 없어 적시성 검증 불가")
            return
        
        try:
            dates = pd.to_datetime(df['draw_date'], errors='coerce').dropna()
            if len(dates) > 0:
                latest_date = dates.max()
                current_date = datetime.now()
                
                # 최신 데이터 확인
                days_since_latest = (current_date - latest_date).days
                
                if days_since_latest > 30:
                    result.add_warning(f"데이터가 오래됨: {days_since_latest}일 전")
                elif days_since_latest > 7:
                    result.add_info(f"데이터 지연: {days_since_latest}일 전")
                else:
                    result.add_info("데이터가 최신임")
                
                result.details['data_freshness_days'] = days_since_latest
        
        except Exception as e:
            result.add_warning(f"적시성 검증 실패: {e}")
    
    def _validate_validity(self, df: pd.DataFrame, result: ValidationResult):
        """타당성 검증"""
        # 로또 게임 규칙 타당성 확인
        
        # 번호 범위 타당성 (1-45)
        valid_range_count = 0
        total_numbers = 0
        
        for col in ['num1', 'num2', 'num3', 'num4', 'num5', 'num6']:
            if col in df.columns:
                numbers = pd.to_numeric(df[col], errors='coerce').dropna()
                valid_numbers = numbers[(numbers >= 1) & (numbers <= 45)]
                valid_range_count += len(valid_numbers)
                total_numbers += len(numbers)
        
        if total_numbers > 0:
            validity_ratio = valid_range_count / total_numbers
            if validity_ratio < 0.99:
                result.add_error(f"번호 타당성 부족: {validity_ratio:.1%}")
            else:
                result.add_info(f"번호 타당성: {validity_ratio:.1%}")
        
        result.details['validity_score'] = validity_ratio if total_numbers > 0 else 0


class PredictionQualityValidator:
    """예측 품질 전문 검증기"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.quality_thresholds = {
            'balance_score': 0.3,
            'diversity_score': 0.4,
            'realism_score': 0.5,
            'consistency_score': 0.6
        }
    
    def validate_prediction_quality(self, numbers: List[int], 
                                  historical_data: pd.DataFrame = None,
                                  prediction_history: List = None) -> ValidationResult:
        """예측 품질 종합 검증"""
        result = ValidationResult()
        
        try:
            # 1. 기본 품질 지표
            quality_scores = self._calculate_quality_scores(numbers)
            result.details['quality_scores'] = quality_scores
            
            # 2. 역사적 비교
            if historical_data is not None:
                historical_comparison = self._compare_with_historical(numbers, historical_data)
                result.details['historical_comparison'] = historical_comparison
            
            # 3. 예측 이력 비교
            if prediction_history:
                diversity_analysis = self._analyze_prediction_diversity(numbers, prediction_history)
                result.details['diversity_analysis'] = diversity_analysis
            
            # 4. 품질 기준 평가
            self._evaluate_quality_standards(quality_scores, result)
            
        except Exception as e:
            result.add_error(f"예측 품질 검증 중 오류: {e}")
        
        return result
    
    def _calculate_quality_scores(self, numbers: List[int]) -> Dict[str, float]:
        """예측 품질 점수 계산"""
        sorted_numbers = sorted(numbers)
        
        # 1. 균형 점수 (구간별 분포)
        sections = {'low': 0, 'mid': 0, 'high': 0}
        for num in numbers:
            if num <= 15:
                sections['low'] += 1
            elif num <= 30:
                sections['mid'] += 1
            else:
                sections['high'] += 1
        
        section_values = list(sections.values())
        balance_score = 1 - (np.var(section_values) / np.mean(section_values)) if np.mean(section_values) > 0 else 0
        
        # 2. 다양성 점수 (번호 간 간격)
        gaps = [sorted_numbers[i+1] - sorted_numbers[i] for i in range(5)]
        gap_diversity = np.std(gaps) / np.mean(gaps) if np.mean(gaps) > 0 else 0
        diversity_score = min(1.0, gap_diversity / 2)  # 정규화
        
        # 3. 현실성 점수 (일반적인 로또 패턴과의 유사성)
        realism_score = self._calculate_realism_score(numbers)
        
        # 4. 일관성 점수 (내부 패턴 일관성)
        consistency_score = self._calculate_consistency_score(numbers)
        
        return {
            'balance_score': balance_score,
            'diversity_score': diversity_score,
            'realism_score': realism_score,
            'consistency_score': consistency_score,
            'overall_score': np.mean([balance_score, diversity_score, realism_score, consistency_score])
        }
    
    def _calculate_realism_score(self, numbers: List[int]) -> float:
        """현실성 점수 계산"""
        score = 0.5  # 기본 점수
        
        # 홀짝 비율 (2-4개가 일반적)
        odd_count = sum(1 for n in numbers if n % 2 == 1)
        if 2 <= odd_count <= 4:
            score += 0.2
        
        # 연속 번호 (1-2쌍이 일반적)
        sorted_numbers = sorted(numbers)
        consecutive_pairs = sum(1 for i in range(5) if sorted_numbers[i+1] - sorted_numbers[i] == 1)
        if consecutive_pairs <= 2:
            score += 0.2
        
        # 번호 합계 (100-150이 일반적)
        total_sum = sum(numbers)
        if 100 <= total_sum <= 150:
            score += 0.1
        
        return min(1.0, score)
    
    def _calculate_consistency_score(self, numbers: List[int]) -> float:
        """일관성 점수 계산"""
        sorted_numbers = sorted(numbers)
        
        # 간격의 일관성
        gaps = [sorted_numbers[i+1] - sorted_numbers[i] for i in range(5)]
        gap_consistency = 1 - (np.std(gaps) / np.mean(gaps)) if np.mean(gaps) > 0 else 0
        
        return max(0, min(1.0, gap_consistency))
    
    def _compare_with_historical(self, numbers: List[int], historical_data: pd.DataFrame) -> Dict:
        """역사적 데이터와 비교"""
        comparison = {}
        
        try:
            number_cols = ['num1', 'num2', 'num3', 'num4', 'num5', 'num6']
            
            # 역사적 번호 빈도
            historical_counts = Counter()
            for col in number_cols:
                if col in historical_data.columns:
                    counts = historical_data[col].dropna().astype(int)
                    historical_counts.update(counts)
            
            # 예측 번호들의 역사적 빈도
            prediction_frequencies = []
            for num in numbers:
                freq = historical_counts.get(num, 0)
                total_appearances = len(historical_data) * 6
                frequency_ratio = freq / total_appearances if total_appearances > 0 else 0
                prediction_frequencies.append(frequency_ratio)
            
            comparison['frequency_analysis'] = {
                'individual_frequencies': list(zip(numbers, prediction_frequencies)),
                'average_frequency': np.mean(prediction_frequencies),
                'frequency_variance': np.var(prediction_frequencies)
            }
            
            # 역사적 평균과 비교
            all_historical = []
            for col in number_cols:
                if col in historical_data.columns:
                    all_historical.extend(historical_data[col].dropna().astype(int).tolist())
            
            if all_historical:
                historical_mean = np.mean(all_historical)
                prediction_mean = np.mean(numbers)
                mean_deviation = abs(prediction_mean - historical_mean)
                
                comparison['statistical_comparison'] = {
                    'historical_mean': historical_mean,
                    'prediction_mean': prediction_mean,
                    'mean_deviation': mean_deviation,
                    'deviation_ratio': mean_deviation / historical_mean if historical_mean > 0 else 0
                }
        
        except Exception as e:
            comparison['error'] = str(e)
        
        return comparison
    
    def _analyze_prediction_diversity(self, numbers: List[int], prediction_history: List) -> Dict:
        """예측 다양성 분석"""
        diversity = {}
        
        try:
            # 최근 예측과의 중복도
            if prediction_history:
                overlaps = []
                for past_prediction in prediction_history[-10:]:  # 최근 10개
                    overlap = len(set(numbers) & set(past_prediction))
                    overlaps.append(overlap)
                
                diversity['recent_overlaps'] = {
                    'average_overlap': np.mean(overlaps),
                    'max_overlap': max(overlaps) if overlaps else 0,
                    'overlap_distribution': Counter(overlaps)
                }
            
            # 번호 분산도
            number_variance = np.var(numbers)
            diversity['number_variance'] = number_variance
            
            # 간격 다양성
            sorted_numbers = sorted(numbers)
            gaps = [sorted_numbers[i+1] - sorted_numbers[i] for i in range(5)]
            gap_diversity = len(set(gaps)) / len(gaps)  # 고유 간격 비율
            diversity['gap_diversity'] = gap_diversity
        
        except Exception as e:
            diversity['error'] = str(e)
        
        return diversity
    
    def _evaluate_quality_standards(self, quality_scores: Dict, result: ValidationResult):
        """품질 기준 평가"""
        for metric, score in quality_scores.items():
            if metric in self.quality_thresholds:
                threshold = self.quality_thresholds[metric]
                
                if score < threshold:
                    result.add_warning(f"{metric} 기준 미달: {score:.3f} < {threshold}")
                else:
                    result.add_info(f"{metric} 기준 통과: {score:.3f}")
        
        # 전체 품질 평가
        overall_score = quality_scores.get('overall_score', 0)
        if overall_score >= 0.7:
            result.add_info("예측 품질이 우수함")
        elif overall_score >= 0.5:
            result.add_info("예측 품질이 보통임")
        else:
            result.add_warning("예측 품질이 낮음")


class ModelPerformanceValidator:
    """모델 성능 검증기"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def validate_model_performance(self, model_results: Dict, 
                                 validation_data: pd.DataFrame = None) -> ValidationResult:
        """모델 성능 검증"""
        result = ValidationResult()
        
        try:
            # 1. 예측 정확도 검증
            if 'accuracy_metrics' in model_results:
                self._validate_accuracy_metrics(model_results['accuracy_metrics'], result)
            
            # 2. 성능 지표 검증
            if 'performance_scores' in model_results:
                self._validate_performance_scores(model_results['performance_scores'], result)
            
            # 3. 백테스팅 결과 검증
            if validation_data is not None:
                backtest_result = self._validate_backtest_performance(model_results, validation_data)
                result.details['backtest_validation'] = backtest_result
            
            # 4. 모델 안정성 검증
            if 'stability_metrics' in model_results:
                self._validate_model_stability(model_results['stability_metrics'], result)
        
        except Exception as e:
            result.add_error(f"모델 성능 검증 중 오류: {e}")
        
        return result
    
    def _validate_accuracy_metrics(self, accuracy_metrics: Dict, result: ValidationResult):
        """정확도 지표 검증"""
        expected_metrics = ['hit_rate', 'partial_hit_rate', 'average_matches']
        
        for metric in expected_metrics:
            if metric not in accuracy_metrics:
                result.add_warning(f"정확도 지표 누락: {metric}")
            else:
                value = accuracy_metrics[metric]
                
                # 합리적인 범위 확인
                if metric == 'hit_rate':
                    if value > 0.1:  # 10% 이상은 비현실적
                        result.add_warning(f"적중률이 비현실적으로 높음: {value:.1%}")
                    elif value > 0:
                        result.add_info(f"적중률: {value:.3%}")
                
                elif metric == 'partial_hit_rate':
                    if 0.1 <= value <= 0.8:  # 10-80%가 합리적
                        result.add_info(f"부분 적중률: {value:.1%}")
                    else:
                        result.add_warning(f"부분 적중률이 비정상적: {value:.1%}")
    
    def _validate_performance_scores(self, performance_scores: Dict, result: ValidationResult):
        """성능 점수 검증"""
        for score_name, score_value in performance_scores.items():
            try:
                numeric_score = float(score_value)
                
                # 0-1 범위 점수들
                if score_name.endswith('_score'):
                    if not (0 <= numeric_score <= 1):
                        result.add_warning(f"{score_name}이 0-1 범위 밖: {numeric_score}")
                
                # 음수 점수 확인
                if numeric_score < 0:
                    result.add_warning(f"{score_name}이 음수: {numeric_score}")
            
            except (ValueError, TypeError):
                result.add_warning(f"{score_name} 값이 숫자가 아님: {score_value}")
    
    def _validate_backtest_performance(self, model_results: Dict, validation_data: pd.DataFrame) -> Dict:
        """백테스팅 성능 검증"""
        backtest_result = {
            'validation_possible': False,
            'performance_metrics': {},
            'issues': []
        }
        
        try:
            if len(validation_data) < 10:
                backtest_result['issues'].append("백테스팅을 위한 데이터 부족")
                return backtest_result
            
            # 간단한 백테스트 수행
            # (실제로는 모델을 다시 훈련해야 하지만, 여기서는 검증만)
            
            backtest_result['validation_possible'] = True
            backtest_result['validation_data_size'] = len(validation_data)
            
            # 모의 성능 지표 (실제 구현에서는 진짜 백테스트)
            backtest_result['performance_metrics'] = {
                'mock_accuracy': 0.15,  # 모의 정확도
                'mock_consistency': 0.6,  # 모의 일관성
                'mock_stability': 0.7     # 모의 안정성
            }
        
        except Exception as e:
            backtest_result['issues'].append(f"백테스팅 검증 오류: {e}")
        
        return backtest_result
    
    def _validate_model_stability(self, stability_metrics: Dict, result: ValidationResult):
        """모델 안정성 검증"""
        stability_checks = {
            'prediction_variance': (0, 0.3),  # 낮을수록 좋음
            'convergence_rate': (0.5, 1.0),   # 높을수록 좋음
            'robustness_score': (0.4, 1.0)    # 높을수록 좋음
        }
        
        for metric, (min_val, max_val) in stability_checks.items():
            if metric in stability_metrics:
                value = stability_metrics[metric]
                try:
                    numeric_value = float(value)
                    if not (min_val <= numeric_value <= max_val):
                        result.add_warning(f"{metric} 기준 미달: {numeric_value:.3f} (기준: {min_val}-{max_val})")
                    else:
                        result.add_info(f"{metric} 기준 통과: {numeric_value:.3f}")
                except (ValueError, TypeError):
                    result.add_warning(f"{metric} 값이 숫자가 아님: {value}")


class ValidationReportGenerator:
    """검증 리포트 생성기"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def generate_comprehensive_report(self, validation_results: Dict[str, ValidationResult]) -> Dict[str, Any]:
        """종합 검증 리포트 생성"""
        report = {
            'timestamp': datetime.now().isoformat(),
            'overall_status': 'unknown',
            'summary': {},
            'detailed_results': {},
            'recommendations': [],
            'critical_issues': [],
            'system_health_score': 0.0
        }
        
        try:
            # 전체 상태 평가
            overall_status = self._determine_overall_status(validation_results)
            report['overall_status'] = overall_status
            
            # 요약 정보
            report['summary'] = self._generate_summary(validation_results)
            
            # 상세 결과
            for validation_type, validation_result in validation_results.items():
                report['detailed_results'][validation_type] = validation_result.to_dict()
            
            # 권장사항 생성
            report['recommendations'] = self._generate_recommendations(validation_results)
            
            # 중요한 문제 식별
            report['critical_issues'] = self._identify_critical_issues(validation_results)
            
            # 시스템 건강 점수
            report['system_health_score'] = self._calculate_system_health_score(validation_results)
        
        except Exception as e:
            report['error'] = f"리포트 생성 중 오류: {e}"
            self.logger.error(f"검증 리포트 생성 오류: {e}")
        
        return report
    
    def _determine_overall_status(self, validation_results: Dict[str, ValidationResult]) -> str:
        """전체 상태 결정"""
        total_errors = sum(len(result.errors) for result in validation_results.values())
        total_warnings = sum(len(result.warnings) for result in validation_results.values())
        
        if total_errors == 0 and total_warnings == 0:
            return 'excellent'
        elif total_errors == 0 and total_warnings <= 5:
            return 'good'
        elif total_errors <= 2 and total_warnings <= 10:
            return 'acceptable'
        elif total_errors <= 5:
            return 'needs_improvement'
        else:
            return 'critical'
    
    def _generate_summary(self, validation_results: Dict[str, ValidationResult]) -> Dict:
        """요약 정보 생성"""
        summary = {
            'total_validations': len(validation_results),
            'passed_validations': sum(1 for result in validation_results.values() if result.is_valid),
            'total_errors': sum(len(result.errors) for result in validation_results.values()),
            'total_warnings': sum(len(result.warnings) for result in validation_results.values()),
            'validation_types': list(validation_results.keys())
        }
        
        summary['pass_rate'] = summary['passed_validations'] / summary['total_validations'] if summary['total_validations'] > 0 else 0
        
        return summary
    
    def _generate_recommendations(self, validation_results: Dict[str, ValidationResult]) -> List[str]:
        """권장사항 생성"""
        recommendations = []
        
        # 오류 기반 권장사항
        for validation_type, result in validation_results.items():
            if result.errors:
                if validation_type == 'data':
                    recommendations.append("데이터 품질 개선이 필요합니다. 원본 데이터를 점검하세요.")
                elif validation_type == 'config':
                    recommendations.append("설정 값을 재검토하고 수정하세요.")
                elif validation_type == 'prediction':
                    recommendations.append("예측 알고리즘을 점검하고 개선하세요.")
                elif validation_type == 'system':
                    recommendations.append("시스템 구성요소들의 연결을 확인하세요.")
        
        # 경고 기반 권장사항
        total_warnings = sum(len(result.warnings) for result in validation_results.values())
        if total_warnings > 10:
            recommendations.append("경고 사항이 많습니다. 시스템 전반적인 점검이 필요합니다.")
        
        # 성능 개선 권장사항
        if 'prediction' in validation_results:
            pred_result = validation_results['prediction']
            if pred_result.details and 'quality_scores' in pred_result.details:
                quality = pred_result.details['quality_scores']
                if quality.get('overall_score', 0) < 0.5:
                    recommendations.append("예측 품질이 낮습니다. 알고리즘 매개변수를 조정하세요.")
        
        return recommendations
    
    def _identify_critical_issues(self, validation_results: Dict[str, ValidationResult]) -> List[str]:
        """중요한 문제 식별"""
        critical_issues = []
        
        critical_keywords = ['누락', '실패', '중복', '범위', '타입', '존재하지 않음']
        
        for validation_type, result in validation_results.items():
            for error in result.errors:
                if any(keyword in error for keyword in critical_keywords):
                    critical_issues.append(f"[{validation_type}] {error}")
        
        return critical_issues
    
    def _calculate_system_health_score(self, validation_results: Dict[str, ValidationResult]) -> float:
        """시스템 건강 점수 계산"""
        total_checks = 0
        passed_checks = 0
        
        for result in validation_results.values():
            # 각 검증의 가중치
            weight = 1.0
            if result.is_valid:
                passed_checks += weight
            
            total_checks += weight
            
            # 경고도 부분적으로 반영
            if result.warnings:
                penalty = min(0.3, len(result.warnings) * 0.05)  # 경고당 5% 차감, 최대 30%
                passed_checks -= penalty
        
        health_score = passed_checks / total_checks if total_checks > 0 else 0
        return max(0, min(1.0, health_score))


# === 유틸리티 함수들 ===

def quick_validate_prediction(numbers: List[int]) -> bool:
    """빠른 예측 유효성 검사"""
    try:
        # 기본 검증만 수행
        if len(numbers) != 6:
            return False
        
        if any(not (1 <= num <= 45) for num in numbers):
            return False
        
        if len(set(numbers)) != 6:
            return False
        
        return True
    
    except Exception:
        return False


def validate_csv_file(file_path: str) -> ValidationResult:
    """CSV 파일 빠른 검증"""
    result = ValidationResult()
    
    try:
        # 파일 존재 확인
        if not Path(file_path).exists():
            result.add_error(f"파일이 존재하지 않음: {file_path}")
            return result
        
        # CSV 로딩 테스트
        df = pd.read_csv(file_path, nrows=5)  # 처음 5행만
        
        # 기본 구조 확인
        required_cols = ['num1', 'num2', 'num3', 'num4', 'num5', 'num6']
        missing_cols = [col for col in required_cols if col not in df.columns]
        
        if missing_cols:
            result.add_error(f"필수 컬럼 누락: {missing_cols}")
        else:
            result.add_info("CSV 파일 기본 구조 정상")
    
    except Exception as e:
        result.add_error(f"CSV 파일 검증 실패: {e}")
    
    return result


def create_validation_summary(validation_results: Dict[str, ValidationResult]) -> str:
    """검증 결과 요약 문자열 생성"""
    total_errors = sum(len(result.errors) for result in validation_results.values())
    total_warnings = sum(len(result.warnings) for result in validation_results.values())
    passed_validations = sum(1 for result in validation_results.values() if result.is_valid)
    
    status_emoji = {
        0: "✅",  # 오류 없음
        1: "⚠️",   # 경고만 있음
        2: "❌"    # 오류 있음
    }
    
    if total_errors == 0:
        status = 0 if total_warnings == 0 else 1
    else:
        status = 2
    
    summary = f"{status_emoji[status]} 검증 요약: "
    summary += f"{passed_validations}/{len(validation_results)} 통과, "
    summary += f"오류 {total_errors}개, 경고 {total_warnings}개"
    
    return summary


# === 모듈 테스트 ===
if __name__ == "__main__":
    # 테스트용 데이터 생성
    print("=== 검증 시스템 테스트 ===")
    
    try:
        # 1. 정상 데이터 테스트
        print("\n1. 정상 데이터 검증 테스트")
        normal_data = []
        for i in range(20):
            numbers = sorted(np.random.choice(range(1, 46), size=6, replace=False))
            row = {'round': 1000 + i}
            for j, num in enumerate(numbers, 1):
                row[f'num{j}'] = num
            normal_data.append(row)
        
        normal_df = pd.DataFrame(normal_data)
        
        data_validator = LottoDataValidator()
        normal_result = data_validator.validate_dataframe(normal_df)
        print(f"정상 데이터 검증: {normal_result.get_summary()}")
        
        # 2. 문제 데이터 테스트
        print("\n2. 문제 데이터 검증 테스트")
        problem_data = []
        for i in range(10):
            if i == 0:  # 중복 번호
                numbers = [1, 1, 3, 4, 5, 6]
            elif i == 1:  # 범위 밖 번호
                numbers = [1, 2, 3, 4, 5, 50]
            elif i == 2:  # 결측값
                numbers = [1, 2, 3, 4, 5, None]
            else:  # 정상
                numbers = sorted(np.random.choice(range(1, 46), size=6, replace=False))
            
            row = {'round': 2000 + i}
            for j, num in enumerate(numbers, 1):
                row[f'num{j}'] = num
            problem_data.append(row)
        
        problem_df = pd.DataFrame(problem_data)
        problem_result = data_validator.validate_dataframe(problem_df)
        print(f"문제 데이터 검증: {problem_result.get_summary()}")
        print(f"오류 목록: {problem_result.errors}")
        
        # 3. 예측 검증 테스트
        print("\n3. 예측 검증 테스트")
        prediction_validator = PredictionValidator()
        
        # 정상 예측
        good_prediction = [5, 12, 18, 25, 33, 41]
        good_result = prediction_validator.validate_prediction(good_prediction)
        print(f"정상 예측 검증: {good_result.get_summary()}")
        
        # 문제 예측
        bad_prediction = [1, 1, 3, 4, 5]  # 중복 + 개수 부족
        bad_result = prediction_validator.validate_prediction(bad_prediction)
        print(f"문제 예측 검증: {bad_result.get_summary()}")
        print(f"오류 목록: {bad_result.errors}")
        
        # 4. 설정 검증 테스트
        print("\n4. 설정 검증 테스트")
        config_validator = ConfigValidator()
        
        # 정상 설정
        good_config = {
            'data_file_path': __file__,  # 현재 파일로 테스트
            'prediction_count': 6,
            'confidence_threshold': 0.5,
            'formula_weights': {'formula1': 0.4, 'formula2': 0.6}
        }
        
        config_result = config_validator.validate_config(good_config)
        print(f"설정 검증: {config_result.get_summary()}")
        
        # 5. 파이프라인 검증 테스트
        print("\n5. 파이프라인 검증 테스트")
        pipeline_validator = PipelineValidator()
        
        # 모의 컴포넌트
        mock_components = {
            'data_loader': "mock_loader",
            'statistics_analyzer': "mock_analyzer",
            'pattern_detector': None,  # 의도적으로 누락
            'formula_engine': "mock_engine"
        }
        
        pipeline_results = pipeline_validator.validate_complete_pipeline(
            normal_df, good_config, mock_components, test_prediction=False
        )
        
        print(f"파이프라인 검증 결과:")
        for validation_type, result in pipeline_results.items():
            print(f"  {validation_type}: {result.get_summary()}")
        
        # 6. 종합 리포트 생성 테스트
        print("\n6. 종합 리포트 테스트")
        report_generator = ValidationReportGenerator()
        
        all_results = {
            'data': normal_result,
            'config': config_result,
            'pipeline': pipeline_results.get('integration', ValidationResult())
        }
        
        comprehensive_report = report_generator.generate_comprehensive_report(all_results)
        print(f"종합 리포트:")
        print(f"  전체 상태: {comprehensive_report['overall_status']}")
        print(f"  시스템 건강 점수: {comprehensive_report['system_health_score']:.3f}")
        print(f"  총 검증: {comprehensive_report['summary']['total_validations']}개")
        print(f"  통과율: {comprehensive_report['summary']['pass_rate']:.1%}")
        
        if comprehensive_report['recommendations']:
            print(f"  권장사항:")
            for rec in comprehensive_report['recommendations'][:3]:
                print(f"    - {rec}")
        
        # 7. 빠른 검증 함수 테스트
        print("\n7. 빠른 검증 함수 테스트")
        
        # 정상 예측
        quick_result1 = quick_validate_prediction([1, 5, 10, 15, 20, 25])
        print(f"빠른 검증 (정상): {quick_result1}")
        
        # 문제 예측
        quick_result2 = quick_validate_prediction([1, 1, 3, 4, 5])
        print(f"빠른 검증 (문제): {quick_result2}")
        
        # CSV 파일 검증
        csv_result = validate_csv_file(__file__)  # 현재 파일로 테스트
        print(f"CSV 파일 검증: {csv_result.get_summary()}")
        
        # 8. 검증 요약 생성 테스트
        print("\n8. 검증 요약 테스트")
        summary_text = create_validation_summary(all_results)
        print(f"검증 요약: {summary_text}")
        
        # 9. 품질 검증기 테스트
        print("\n9. 예측 품질 검증 테스트")
        quality_validator = PredictionQualityValidator()
        
        # 고품질 예측
        high_quality_prediction = [3, 12, 18, 24, 33, 42]  # 균형 잡힌 예측
        quality_result = quality_validator.validate_prediction_quality(
            high_quality_prediction, 
            historical_data=normal_df
        )
        print(f"고품질 예측 검증: {quality_result.get_summary()}")
        if quality_result.details and 'quality_scores' in quality_result.details:
            scores = quality_result.details['quality_scores']
            print(f"  품질 점수: {scores['overall_score']:.3f}")
            print(f"  균형 점수: {scores['balance_score']:.3f}")
            print(f"  다양성 점수: {scores['diversity_score']:.3f}")
        
        # 저품질 예측
        low_quality_prediction = [1, 2, 3, 4, 5, 6]  # 연속 번호
        quality_result2 = quality_validator.validate_prediction_quality(low_quality_prediction)
        print(f"저품질 예측 검증: {quality_result2.get_summary()}")
        
        # 10. 모델 성능 검증기 테스트
        print("\n10. 모델 성능 검증 테스트")
        performance_validator = ModelPerformanceValidator()
        
        mock_model_results = {
            'accuracy_metrics': {
                'hit_rate': 0.05,
                'partial_hit_rate': 0.25,
                'average_matches': 1.2
            },
            'performance_scores': {
                'overall_score': 0.65,
                'consistency_score': 0.7,
                'stability_score': 0.8
            },
            'stability_metrics': {
                'prediction_variance': 0.15,
                'convergence_rate': 0.85,
                'robustness_score': 0.6
            }
        }
        
        performance_result = performance_validator.validate_model_performance(
            mock_model_results, 
            validation_data=normal_df
        )
        print(f"모델 성능 검증: {performance_result.get_summary()}")
        
        if performance_result.details:
            print(f"  백테스트 가능: {performance_result.details.get('backtest_validation', {}).get('validation_possible', False)}")
        
        print("\n✅ 모든 검증 시스템 테스트 완료")
        print("\n📋 검증 시스템 사용 가이드:")
        print("1. LottoDataValidator: CSV 데이터 검증")
        print("2. PredictionValidator: 예측 결과 검증") 
        print("3. ConfigValidator: 설정 값 검증")
        print("4. SystemIntegrityValidator: 시스템 무결성 검증")
        print("5. PipelineValidator: 전체 파이프라인 검증")
        print("6. PredictionQualityValidator: 예측 품질 전문 검증")
        print("7. ModelPerformanceValidator: 모델 성능 검증")
        print("8. ValidationReportGenerator: 종합 리포트 생성")
        print("\n🔧 유틸리티 함수:")
        print("- quick_validate_prediction(): 빠른 예측 검증")
        print("- validate_csv_file(): CSV 파일 빠른 검증")
        print("- create_validation_summary(): 검증 요약 생성")

    except Exception as e:
        print(f"❌ 테스트 중 오류: {e}")
        import traceback
        traceback.print_exc()
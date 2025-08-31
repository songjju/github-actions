"""
파일명: src/data_processing/data_cleaner.py
목적: 로또 데이터 정제 및 표준화
작성자: AI Assistant
작성일: 2025-08-31
버전: 1.0

주요 클래스/함수:
- LottoDataCleaner: 로또 데이터 정제 클래스
- clean_winning_numbers: 당첨번호 정제
- normalize_prize_amounts: 당첨금액 정규화
- handle_missing_data: 결측치 처리
"""

import pandas as pd
import numpy as np
import re
from typing import Dict, List, Tuple, Any, Optional, Union
import logging
from datetime import datetime, timedelta
from pathlib import Path
import sys
import warnings

# 상위 디렉토리 imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.utils.config import Config, LottoConstants

warnings.filterwarnings('ignore')

class LottoDataCleaner:
    """로또 데이터 정제 클래스"""
    
    def __init__(self, strict_mode: bool = True):
        """
        초기화
        
        Args:
            strict_mode (bool): 엄격 모드 (오류 시 예외 발생)
        """
        self.strict_mode = strict_mode
        self.logger = self._setup_logger()
        self.cleaning_stats = {
            'total_rows': 0,
            'cleaned_rows': 0,
            'removed_rows': 0,
            'corrected_values': 0,
            'issues_found': []
        }
        
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
    
    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        전체 데이터 정제 파이프라인
        
        Args:
            df (pd.DataFrame): 원본 데이터프레임
            
        Returns:
            pd.DataFrame: 정제된 데이터프레임
        """
        self.logger.info("데이터 정제 시작")
        self.cleaning_stats['total_rows'] = len(df)
        
        cleaned_df = df.copy()
        
        try:
            # 1. 기본 정제
            cleaned_df = self._basic_cleaning(cleaned_df)
            
            # 2. 당첨번호 정제
            cleaned_df = self._clean_winning_numbers(cleaned_df)
            
            # 3. 당첨금액 정제
            cleaned_df = self._clean_prize_amounts(cleaned_df)
            
            # 4. 날짜 정제
            cleaned_df = self._clean_dates(cleaned_df)
            
            # 5. 당첨자 수 정제
            cleaned_df = self._clean_winner_counts(cleaned_df)
            
            # 6. 결측치 처리
            cleaned_df = self._handle_missing_data(cleaned_df)
            
            # 7. 데이터 타입 최적화
            cleaned_df = self._optimize_data_types(cleaned_df)
            
            # 8. 최종 검증
            cleaned_df = self._final_validation(cleaned_df)
            
            self.cleaning_stats['cleaned_rows'] = len(cleaned_df)
            self.cleaning_stats['removed_rows'] = self.cleaning_stats['total_rows'] - len(cleaned_df)
            
            self.logger.info(f"데이터 정제 완료: {len(cleaned_df)}행 (제거: {self.cleaning_stats['removed_rows']}행)")
            
            return cleaned_df
            
        except Exception as e:
            self.logger.error(f"데이터 정제 중 오류: {e}")
            if self.strict_mode:
                raise
            return df
    
    def debug_data_types(self, df: pd.DataFrame) -> Dict[str, Any]:
        """데이터 타입 디버깅 정보 수집"""
        debug_info = {
            'columns': list(df.columns),
            'dtypes': df.dtypes.to_dict(),
            'problematic_columns': {},
            'sample_values': {}
        }
        
        for col in df.columns:
            # 각 컬럼의 고유한 타입들 확인
            unique_types = set()
            sample_values = []
            
            for idx, value in df[col].dropna().head(10).items():
                value_type = type(value).__name__
                unique_types.add(value_type)
                sample_values.append(f"{value} ({value_type})")
                
                # 문제가 될 수 있는 타입 식별
                if isinstance(value, (list, dict, tuple, set)):
                    debug_info['problematic_columns'][col] = {
                        'type': value_type,
                        'sample_value': str(value)[:100],
                        'index': idx
                    }
            
            debug_info['sample_values'][col] = sample_values
            
            if len(unique_types) > 1:
                debug_info['problematic_columns'][col] = {
                    'issue': 'mixed_types',
                    'types': list(unique_types)
                }
        
        return debug_info

    def _basic_cleaning(self, df: pd.DataFrame) -> pd.DataFrame:
        """기본 정제 작업 (unhashable type 오류 해결)"""
        self.logger.info("기본 정제 시작")
        
        # 컬럼명 정리
        df.columns = df.columns.str.strip()
        
        # 리스트 타입 데이터 확인 및 변환
        self.logger.info("데이터 타입 확인 및 변환")
        for col in df.columns:
            # 각 컬럼의 첫 번째 유효한 값의 타입 확인
            first_valid_idx = df[col].first_valid_index()
            if first_valid_idx is not None:
                first_value = df.loc[first_valid_idx, col]
                
                # 리스트 타입인 경우 문자열로 변환
                if isinstance(first_value, list):
                    self.logger.warning(f"컬럼 '{col}'에서 list 타입 발견, 문자열로 변환")
                    df[col] = df[col].apply(lambda x: str(x) if isinstance(x, list) else x)
                    self.cleaning_stats['corrected_values'] += 1
                
                # 딕셔너리 타입인 경우도 문자열로 변환
                elif isinstance(first_value, dict):
                    self.logger.warning(f"컬럼 '{col}'에서 dict 타입 발견, 문자열로 변환")
                    df[col] = df[col].apply(lambda x: str(x) if isinstance(x, dict) else x)
                    self.cleaning_stats['corrected_values'] += 1
        
        # 전체 DataFrame에서 리스트/딕셔너리 타입 일괄 변환
        try:
            for col in df.columns:
                if df[col].apply(lambda x: isinstance(x, (list, dict))).any():
                    self.logger.warning(f"컬럼 '{col}'에서 복합 타입 발견, 변환 중")
                    df[col] = df[col].apply(lambda x: str(x) if isinstance(x, (list, dict, tuple, set)) else x)
        except Exception as e:
            self.logger.error(f"데이터 타입 변환 중 오류: {e}")
        
        # 이제 안전하게 중복 행 제거
        try:
            initial_count = len(df)
            df = df.drop_duplicates()
            removed_duplicates = initial_count - len(df)
            
            if removed_duplicates > 0:
                self.cleaning_stats['issues_found'].append(f"중복 행 {removed_duplicates}개 제거")
                self.logger.info(f"중복 행 {removed_duplicates}개 제거")
                
        except Exception as e:
            self.logger.error(f"중복 제거 중 오류: {e}")
            # 중복 제거에 실패하면 경고만 하고 계속 진행
            self.cleaning_stats['issues_found'].append(f"중복 제거 실패: {str(e)}")
        
        # 빈 행 제거
        df = df.dropna(how='all')
        
        return df
    
    def _clean_winning_numbers(self, df: pd.DataFrame) -> pd.DataFrame:
        """당첨번호 정제"""
        self.logger.info("당첨번호 정제 시작")
        
        # 당첨번호 컬럼 식별
        number_columns = []
        for col in df.columns:
            if any(keyword in col.lower() for keyword in ['num', '번호', 'number']):
                if not any(exclude in col.lower() for exclude in ['prize', '금액', 'winner', '당첨자']):
                    number_columns.append(col)
        
        # 정확한 컬럼명으로 매핑
        expected_number_cols = ['num1', 'num2', 'num3', 'num4', 'num5', 'num6', 'bonus']
        actual_number_cols = [col for col in df.columns if col in expected_number_cols]
        
        if actual_number_cols:
            number_columns = actual_number_cols
        
        self.logger.info(f"당첨번호 컬럼: {number_columns}")
        
        for col in number_columns:
            if col in df.columns:
                # 문자열에서 숫자 추출
                df[col] = df[col].astype(str).str.extract(r'(\d+)').astype(float)
                
                # 유효 범위 확인 (1-45)
                invalid_mask = (df[col] < 1) | (df[col] > 45) | df[col].isna()
                invalid_count = invalid_mask.sum()
                
                if invalid_count > 0:
                    self.cleaning_stats['issues_found'].append(f"{col}: 유효하지 않은 값 {invalid_count}개")
                    
                    if self.strict_mode:
                        # 엄격 모드에서는 해당 행 제거
                        df = df[~invalid_mask]
                    else:
                        # 관대 모드에서는 중앙값으로 대체
                        median_val = df[col].median()
                        df.loc[invalid_mask, col] = median_val
                        self.cleaning_stats['corrected_values'] += invalid_count
                
                # 정수형으로 변환
                df[col] = df[col].astype('Int64')
        
        return df
    
    def _clean_prize_amounts(self, df: pd.DataFrame) -> pd.DataFrame:
        """당첨금액 정제"""
        self.logger.info("당첨금액 정제 시작")
        
        # 당첨금액 컬럼 식별
        prize_columns = []
        for col in df.columns:
            if any(keyword in col.lower() for keyword in ['prize', '금액', 'amount', 'money']):
                prize_columns.append(col)
        
        for col in prize_columns:
            if col in df.columns:
                original_values = df[col].copy()
                
                # 문자열 처리
                df[col] = df[col].astype(str)
                
                # 쉼표, 원화 표시 제거
                df[col] = df[col].str.replace(',', '').str.replace('원', '').str.replace('₩', '')
                
                # 한글 숫자 단위 처리 (억, 만 등)
                df[col] = df[col].apply(self._convert_korean_number)
                
                # 숫자 추출
                df[col] = pd.to_numeric(df[col], errors='coerce')
                
                # 음수 값 처리
                negative_mask = df[col] < 0
                if negative_mask.sum() > 0:
                    df.loc[negative_mask, col] = 0
                    self.cleaning_stats['corrected_values'] += negative_mask.sum()
                
                # 비현실적으로 큰 값 처리 (100억 이상)
                too_large_mask = df[col] > 10_000_000_000
                if too_large_mask.sum() > 0:
                    median_val = df[col].median()
                    df.loc[too_large_mask, col] = median_val
                    self.cleaning_stats['corrected_values'] += too_large_mask.sum()
        
        return df
    
    def _convert_korean_number(self, value_str: str) -> str:
        """한글 숫자 단위를 숫자로 변환"""
        if pd.isna(value_str) or value_str == 'nan':
            return '0'
        
        value_str = str(value_str).strip()
        
        # 억 단위 처리
        if '억' in value_str:
            parts = value_str.split('억')
            if len(parts) == 2:
                try:
                    left = float(parts[0].replace(',', '')) if parts[0] else 0
                    right = float(parts[1].replace(',', '').replace('만', '0000')) if parts[1] else 0
                    return str(int(left * 100000000 + right))
                except:
                    pass
        
        # 만 단위 처리
        if '만' in value_str:
            parts = value_str.split('만')
            if len(parts) == 2:
                try:
                    left = float(parts[0].replace(',', '')) if parts[0] else 0
                    right = float(parts[1].replace(',', '')) if parts[1] else 0
                    return str(int(left * 10000 + right))
                except:
                    pass
        
        # 일반 숫자 처리
        cleaned = re.sub(r'[^\d.]', '', value_str)
        return cleaned if cleaned else '0'
    
    def _clean_dates(self, df: pd.DataFrame) -> pd.DataFrame:
        """날짜 데이터 정제"""
        self.logger.info("날짜 정제 시작")
        
        # 날짜 컬럼 식별
        date_columns = []
        for col in df.columns:
            if any(keyword in col.lower() for keyword in ['date', '날짜', 'draw']):
                date_columns.append(col)
        
        for col in date_columns:
            if col in df.columns:
                # 날짜 형식 통일
                df[col] = pd.to_datetime(df[col], errors='coerce')
                
                # 유효하지 않은 날짜 확인
                invalid_dates = df[col].isna().sum()
                if invalid_dates > 0:
                    self.cleaning_stats['issues_found'].append(f"{col}: 유효하지 않은 날짜 {invalid_dates}개")
                
                # 미래 날짜 확인
                future_mask = df[col] > pd.Timestamp.now()
                future_count = future_mask.sum()
                if future_count > 0:
                    self.cleaning_stats['issues_found'].append(f"{col}: 미래 날짜 {future_count}개")
                    if self.strict_mode:
                        df = df[~future_mask]
                
                # 너무 과거 날짜 확인 (1945년 이전)
                too_old_mask = df[col] < pd.Timestamp('1945-01-01')
                too_old_count = too_old_mask.sum()
                if too_old_count > 0:
                    self.cleaning_stats['issues_found'].append(f"{col}: 너무 과거 날짜 {too_old_count}개")
                    if self.strict_mode:
                        df = df[~too_old_mask]
        
        return df
    
    def _clean_winner_counts(self, df: pd.DataFrame) -> pd.DataFrame:
        """당첨자 수 정제"""
        self.logger.info("당첨자 수 정제 시작")
        
        # 당첨자 수 컬럼 식별
        winner_columns = []
        for col in df.columns:
            if any(keyword in col.lower() for keyword in ['winner', '당첨자', 'count']):
                if not any(exclude in col.lower() for exclude in ['prize', '금액', 'amount']):
                    winner_columns.append(col)
        
        for col in winner_columns:
            if col in df.columns:
                # 숫자로 변환
                df[col] = pd.to_numeric(df[col], errors='coerce')
                
                # 음수 값 처리
                negative_mask = df[col] < 0
                if negative_mask.sum() > 0:
                    df.loc[negative_mask, col] = 0
                    self.cleaning_stats['corrected_values'] += negative_mask.sum()
                
                # 비현실적으로 큰 값 처리 (1억 명 이상)
                too_large_mask = df[col] > 100_000_000
                if too_large_mask.sum() > 0:
                    median_val = df[col].median()
                    df.loc[too_large_mask, col] = median_val
                    self.cleaning_stats['corrected_values'] += too_large_mask.sum()
                
                # 정수형으로 변환
                df[col] = df[col].astype('Int64')
        
        return df
    
    def _handle_missing_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """결측치 처리"""
        self.logger.info("결측치 처리 시작")
        
        # 당첨번호 결측치 처리
        number_cols = [col for col in df.columns if col in ['num1', 'num2', 'num3', 'num4', 'num5', 'num6', 'bonus']]
        
        for col in number_cols:
            if col in df.columns:
                missing_count = df[col].isna().sum()
                if missing_count > 0:
                    self.cleaning_stats['issues_found'].append(f"{col}: 결측치 {missing_count}개")
                    
                    if self.strict_mode:
                        # 당첨번호 결측치가 있는 행은 삭제
                        df = df.dropna(subset=[col])
                    else:
                        # 해당 번호의 최빈값으로 대체
                        mode_val = df[col].mode()[0] if not df[col].mode().empty else 1
                        df[col] = df[col].fillna(mode_val)
                        self.cleaning_stats['corrected_values'] += missing_count
        
        # 기타 컬럼 결측치 처리
        for col in df.columns:
            if col not in number_cols:
                missing_count = df[col].isna().sum()
                if missing_count > 0:
                    if df[col].dtype in ['int64', 'Int64', 'float64']:
                        # 숫자형은 중앙값으로 대체
                        df[col] = df[col].fillna(df[col].median())
                    else:
                        # 문자형은 최빈값으로 대체
                        mode_val = df[col].mode()[0] if not df[col].mode().empty else 'Unknown'
                        df[col] = df[col].fillna(mode_val)
                    
                    self.cleaning_stats['corrected_values'] += missing_count
        
        return df
    
    def _optimize_data_types(self, df: pd.DataFrame) -> pd.DataFrame:
        """데이터 타입 최적화"""
        self.logger.info("데이터 타입 최적화 시작")
        
        # 당첨번호는 작은 정수형 사용
        number_cols = [col for col in df.columns if col in ['num1', 'num2', 'num3', 'num4', 'num5', 'num6', 'bonus']]
        for col in number_cols:
            if col in df.columns:
                df[col] = df[col].astype('int8')
        
        # 회차번호 최적화
        if 'round' in df.columns:
            df['round'] = df['round'].astype('int16')
        
        # 연도 최적화
        if 'year' in df.columns:
            df['year'] = df['year'].astype('int16')
        
        # 당첨자 수 최적화
        winner_cols = [col for col in df.columns if 'winner' in col.lower()]
        for col in winner_cols:
            if col in df.columns:
                max_val = df[col].max() if not df[col].empty else 0
                if max_val < 2**15:
                    df[col] = df[col].astype('int16')
                elif max_val < 2**31:
                    df[col] = df[col].astype('int32')
        
        return df
    
    def _final_validation(self, df: pd.DataFrame) -> pd.DataFrame:
        """최종 검증"""
        self.logger.info("최종 검증 시작")
        
        # 당첨번호 유효성 재검증
        number_cols = [col for col in df.columns if col in ['num1', 'num2', 'num3', 'num4', 'num5', 'num6', 'bonus']]
        
        if len(number_cols) >= 6:  # 최소 6개 당첨번호가 있어야 함
            # 각 행에서 당첨번호 6개가 모두 다른지 확인
            main_cols = [col for col in number_cols if col != 'bonus'][:6]
            
            for idx, row in df.iterrows():
                numbers = [row[col] for col in main_cols if pd.notna(row[col])]
                
                # 중복 번호 확인
                if len(numbers) != len(set(numbers)):
                    self.cleaning_stats['issues_found'].append(f"행 {idx}: 중복 당첨번호")
                    if self.strict_mode:
                        df = df.drop(idx)
                
                # 범위 확인
                invalid_numbers = [n for n in numbers if not (1 <= n <= 45)]
                if invalid_numbers:
                    self.cleaning_stats['issues_found'].append(f"행 {idx}: 범위 외 번호 {invalid_numbers}")
                    if self.strict_mode:
                        df = df.drop(idx)
        
        # 인덱스 재설정
        df = df.reset_index(drop=True)
        
        return df
    
    def get_cleaning_report(self) -> Dict[str, Any]:
        """정제 결과 리포트"""
        return {
            'cleaning_summary': {
                'total_input_rows': self.cleaning_stats['total_rows'],
                'final_output_rows': self.cleaning_stats['cleaned_rows'],
                'removed_rows': self.cleaning_stats['removed_rows'],
                'corrected_values': self.cleaning_stats['corrected_values'],
                'success_rate': self.cleaning_stats['cleaned_rows'] / self.cleaning_stats['total_rows'] if self.cleaning_stats['total_rows'] > 0 else 0
            },
            'issues_found': self.cleaning_stats['issues_found'],
            'recommendations': self._generate_recommendations()
        }
    
    def _generate_recommendations(self) -> List[str]:
        """개선 권장사항 생성"""
        recommendations = []
        
        if self.cleaning_stats['removed_rows'] > self.cleaning_stats['total_rows'] * 0.1:
            recommendations.append("10% 이상의 데이터가 제거되었습니다. 데이터 품질을 확인하세요.")
        
        if self.cleaning_stats['corrected_values'] > self.cleaning_stats['total_rows']:
            recommendations.append("많은 값이 수정되었습니다. 원본 데이터의 형식을 확인하세요.")
        
        if len(self.cleaning_stats['issues_found']) > 10:
            recommendations.append("다양한 데이터 품질 문제가 발견되었습니다. 데이터 수집 과정을 점검하세요.")
        
        if not recommendations:
            recommendations.append("데이터 품질이 양호합니다.")
        
        return recommendations
    
    def clean_specific_columns(self, df: pd.DataFrame, column_rules: Dict[str, Dict]) -> pd.DataFrame:
        """
        특정 컬럼에 대한 맞춤 정제
        
        Args:
            df: 데이터프레임
            column_rules: 컬럼별 정제 규칙
                예: {
                    'num1': {'min': 1, 'max': 45, 'fill_method': 'median'},
                    'prize': {'remove_chars': [',', '원'], 'convert_units': True}
                }
        """
        for col, rules in column_rules.items():
            if col in df.columns:
                # 범위 제한
                if 'min' in rules and 'max' in rules:
                    df[col] = df[col].clip(lower=rules['min'], upper=rules['max'])
                
                # 문자 제거
                if 'remove_chars' in rules:
                    for char in rules['remove_chars']:
                        df[col] = df[col].astype(str).str.replace(char, '')
                
                # 단위 변환
                if rules.get('convert_units', False):
                    df[col] = df[col].apply(self._convert_korean_number)
                
                # 결측치 처리 방법
                fill_method = rules.get('fill_method', 'median')
                if fill_method == 'median':
                    df[col] = df[col].fillna(df[col].median())
                elif fill_method == 'mode':
                    mode_val = df[col].mode()[0] if not df[col].mode().empty else 0
                    df[col] = df[col].fillna(mode_val)
                elif fill_method == 'zero':
                    df[col] = df[col].fillna(0)
        
        return df


# === 모듈 테스트 ===
if __name__ == "__main__":
    # 테스트용 더미 데이터 생성
    test_data = {
        'round': [1, 2, 3, 4, 5],
        'num1': [1, 2, '3', 4, None],
        'num2': [7, 8, 9, 10, 11],
        'num3': [14, 15, 16, 17, 18],
        'num4': [21, 22, 23, 24, 25],
        'num5': [28, 29, 30, 31, 32],
        'num6': [35, 36, 37, 38, 39],
        'bonus': [42, 43, 44, 45, 1],
        '1st_prize': ['1,000,000,000원', '2억원', '15억 5천만원', '800000000', None],
        '1st_winners': [1, 2, 0, 1, 3],
        'draw_date': ['2023-01-07', '2023-01-14', '2023-01-21', '2023-01-28', '2023-02-04']
    }
    
    test_df = pd.DataFrame(test_data)
    
    print("=== 데이터 정제기 테스트 ===")
    print(f"원본 데이터:\n{test_df}")
    print(f"원본 데이터 타입:\n{test_df.dtypes}")
    
    try:
        # 정제기 초기화
        cleaner = LottoDataCleaner(strict_mode=False)
        
        # 데이터 정제 수행
        cleaned_df = cleaner.clean_data(test_df)
        
        print(f"\n정제된 데이터:\n{cleaned_df}")
        print(f"\n정제된 데이터 타입:\n{cleaned_df.dtypes}")
        
        # 정제 리포트
        report = cleaner.get_cleaning_report()
        print(f"\n=== 정제 리포트 ===")
        print(f"입력 행 수: {report['cleaning_summary']['total_input_rows']}")
        print(f"출력 행 수: {report['cleaning_summary']['final_output_rows']}")
        print(f"제거된 행 수: {report['cleaning_summary']['removed_rows']}")
        print(f"수정된 값 수: {report['cleaning_summary']['corrected_values']}")
        print(f"성공률: {report['cleaning_summary']['success_rate']:.2%}")
        
        if report['issues_found']:
            print(f"\n발견된 문제들:")
            for issue in report['issues_found']:
                print(f"  - {issue}")
        
        if report['recommendations']:
            print(f"\n권장사항:")
            for rec in report['recommendations']:
                print(f"  - {rec}")
        
        print("\n✅ 데이터 정제기 테스트 완료")
        
    except Exception as e:
        print(f"❌ 테스트 중 오류: {e}")
        import traceback
        traceback.print_exc()
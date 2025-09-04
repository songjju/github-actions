"""
파일명: src/data_processing/data_loader.py
목적: 로또 CSV 데이터 로딩 및 전처리 엔진
작성자: AI Assistant
작성일: 2025-08-30
버전: 1.0

주요 클래스/함수:
- LottoDataLoader: CSV 데이터 로딩 및 파싱 클래스
- validate_data_integrity(): 데이터 무결성 검증 함수
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Tuple, Optional, Any
import re
import logging
from pathlib import Path
import sys
import os

# 상위 디렉토리의 config 임포트를 위한 경로 설정
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.utils.config import Config, LottoConstants, validate_lotto_numbers

class LottoDataLoader:
    """로또 데이터 로딩 및 전처리 클래스"""
    
    def __init__(self, csv_path: str):
        """
        초기화
        
        Args:
            csv_path (str): CSV 파일 경로
        """
        self.csv_path = Path(csv_path)
        self.raw_data = None
        self.processed_data = None
        self.logger = self._setup_logger()
        
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
    
    def load_and_preprocess(self) -> pd.DataFrame:
        """
        CSV 파일 로딩 및 전처리 수행
        
        Returns:
            pd.DataFrame: 전처리된 로또 데이터
            
        Raises:
            FileNotFoundError: CSV 파일이 존재하지 않을 때
            ValueError: 데이터 형식이 올바르지 않을 때
        """
        try:
            self.logger.info(f"CSV 파일 로딩 시작: {self.csv_path}")
            
            # 1. CSV 파일 존재 확인
            if not self.csv_path.exists():
                raise FileNotFoundError(f"CSV 파일을 찾을 수 없습니다: {self.csv_path}")
            
            # 2. CSV 파일 로딩 (복잡한 형식 처리)
            self.raw_data = self._load_complex_csv()
            
            # 3. 데이터 전처리
            self.processed_data = self._preprocess_data()
            
            # 4. 데이터 검증
            self._validate_processed_data()
            
            self.logger.info(f"데이터 로딩 완료: {len(self.processed_data)}개 행")
            return self.processed_data
            
        except Exception as e:
            self.logger.error(f"데이터 로딩 중 오류 발생: {e}")
            raise

    def _load_complex_csv(self) -> pd.DataFrame:
        """
        CSV 파일 로딩 (새로운 깔끔한 구조)
        
        Returns:
            pd.DataFrame: 로딩된 원본 데이터
        """
        try:
            # Papa Parse를 사용한 간단한 로딩
            import pandas as pd
            
            # CSV 파일을 pandas로 직접 로딩
            df = pd.read_csv(self.csv_path, encoding='utf-8')
            
            self.logger.info(f"원본 데이터 로딩 완료: {len(df)}개 행, {len(df.columns)}개 컬럼")
            self.logger.info(f"컬럼 목록: {list(df.columns)}")
            
            return df
            
        except Exception as e:
            self.logger.error(f"CSV 로딩 중 오류: {e}")
            raise

    
    def _preprocess_data(self) -> pd.DataFrame:
        """
        데이터 전처리 수행 (CSV 구조에 맞춰 수정)
        
        Returns:
            pd.DataFrame: 전처리된 데이터
        """
        df = self.raw_data.copy()
        
        try:
            self.logger.info(f"원본 데이터 전처리 시작: {len(df)}행")
            
            # 데이터가 비어있는지 확인
            if df.empty:
                raise ValueError("원본 데이터가 비어있습니다.")
            
            # 필수 컬럼 존재 확인
            required_columns = ['num1', 'num2', 'num3', 'num4', 'num5', 'num6']
            missing_columns = [col for col in required_columns if col not in df.columns]
            
            if missing_columns:
                self.logger.error(f"필수 컬럼 누락: {missing_columns}")
                self.logger.error(f"실제 컬럼: {list(df.columns)}")
                raise ValueError(f"필수 컬럼이 누락되었습니다: {missing_columns}")
            
            # 새로운 DataFrame 구조로 변환
            processed_rows = []
            skipped_rows = 0
            
            for idx, row in df.iterrows():
                try:
                    # 당첨번호 6개 추출
                    winning_numbers = [
                        int(row['num1']), int(row['num2']), int(row['num3']),
                        int(row['num4']), int(row['num5']), int(row['num6'])
                    ]
                    
                    # 번호 유효성 검사
                    if not all(1 <= num <= 45 for num in winning_numbers):
                        self.logger.warning(f"행 {idx}: 유효하지 않은 번호 범위 {winning_numbers}")
                        skipped_rows += 1
                        continue
                    
                    # 중복 번호 확인
                    if len(set(winning_numbers)) != 6:
                        self.logger.warning(f"행 {idx}: 중복 번호 발견 {winning_numbers}")
                        skipped_rows += 1
                        continue
                    
                    # 보너스 번호 처리
                    bonus_number = None
                    if 'bonus' in df.columns and pd.notna(row['bonus']):
                        try:
                            bonus_number = int(row['bonus'])
                            if not (1 <= bonus_number <= 45):
                                self.logger.warning(f"행 {idx}: 유효하지 않은 보너스 번호 {bonus_number}")
                                bonus_number = None
                        except (ValueError, TypeError):
                            self.logger.warning(f"행 {idx}: 보너스 번호 변환 실패")
                            bonus_number = None
                    
                    # 추가 통계 계산
                    number_sum = sum(winning_numbers)
                    odd_count = sum(1 for num in winning_numbers if num % 2 == 1)
                    even_count = 6 - odd_count
                    low_count = sum(1 for num in winning_numbers if num <= 15)
                    mid_count = sum(1 for num in winning_numbers if 16 <= num <= 30)
                    high_count = sum(1 for num in winning_numbers if num >= 31)
                    
                    # 처리된 행 데이터 생성
                    processed_row = {
                        # 기본 정보
                        'round_number': int(row['round']) if 'round' in df.columns and pd.notna(row['round']) else idx + 1,
                        'year': int(row['year']) if 'year' in df.columns and pd.notna(row['year']) else 2025,
                        'draw_date': str(row['draw_date']) if 'draw_date' in df.columns and pd.notna(row['draw_date']) else '',
                        
                        # 당첨번호 (개별)
                        'number_1': winning_numbers[0],
                        'number_2': winning_numbers[1], 
                        'number_3': winning_numbers[2],
                        'number_4': winning_numbers[3],
                        'number_5': winning_numbers[4],
                        'number_6': winning_numbers[5],
                        
                        # 당첨번호 (리스트)
                        'winning_numbers': winning_numbers,
                        'bonus_number': bonus_number,
                        
                        # 통계 정보
                        'number_sum': number_sum,
                        'odd_count': odd_count,
                        'even_count': even_count,
                        'low_count': low_count,   # 1-15
                        'mid_count': mid_count,   # 16-30  
                        'high_count': high_count, # 31-45
                        
                        # 당첨 정보 (있다면)
                        '1st_winners': row.get('1st_winners', 0),
                        '1st_prize': str(row.get('1st_prize', '')) if pd.notna(row.get('1st_prize')) else '',
                    }
                    
                    processed_rows.append(processed_row)
                    
                except Exception as e:
                    self.logger.warning(f"행 {idx} 처리 중 오류 (건너뜀): {e}")
                    skipped_rows += 1
                    continue
            
            # DataFrame 생성
            if not processed_rows:
                raise ValueError("처리된 유효한 데이터가 없습니다.")
            
            processed_df = pd.DataFrame(processed_rows)
            
            # 정렬 (최신 회차가 위로)
            if 'round_number' in processed_df.columns:
                processed_df = processed_df.sort_values('round_number', ascending=False).reset_index(drop=True)
            
            self.logger.info(f"데이터 전처리 완료: {len(processed_df)}행 처리, {skipped_rows}행 건너뜀")
            
            return processed_df
            
        except Exception as e:
            self.logger.error(f"데이터 전처리 중 오류: {e}")
            raise

    # 추가: load_sample_data 메서드도 추가
    def load_sample_data(self, n_samples: int = 100) -> pd.DataFrame:
        """
        샘플 데이터 로딩
        
        Args:
            n_samples: 로딩할 샘플 수
            
        Returns:
            pd.DataFrame: 샘플 데이터
        """
        if self.processed_data is None:
            self.load_and_preprocess()
        
        if self.processed_data is None or self.processed_data.empty:
            return pd.DataFrame()
        
        return self.processed_data.head(n_samples)
        
    def _extract_round_number(self, row: pd.Series) -> Optional[int]:
        """
        행에서 회차 번호 추출 (새로운 구조에서는 'round' 컬럼 직접 사용)
            
        Args:
            row (pd.Series): 데이터 행
                
        Returns:
            Optional[int]: 회차 번호
        """
        try:
            return int(row['round'])
        except:
            return None
    
    def _extract_date_info(self, row: pd.Series) -> Optional[str]:
        """
        행에서 날짜 정보 추출 (새로운 구조에서는 'draw_date' 컬럼 직접 사용)
        
        Args:
            row (pd.Series): 데이터 행
            
        Returns:
            Optional[str]: 날짜 정보
        """
        try:
            return str(row['draw_date'])
        except:
            return None
    
    def _validate_processed_data(self):
        """처리된 데이터의 무결성 검증"""
        if self.processed_data is None or self.processed_data.empty:
            raise ValueError("처리된 데이터가 비어있습니다.")
        
        required_columns = [
            'round_number', 'number_1', 'number_2', 'number_3',
            'number_4', 'number_5', 'number_6', 'winning_numbers'
        ]
        
        missing_columns = [col for col in required_columns if col not in self.processed_data.columns]
        if missing_columns:
            raise ValueError(f"필수 컬럼이 누락되었습니다: {missing_columns}")
        
        # 당첨번호 유효성 검증
        invalid_count = 0
        for idx, row in self.processed_data.iterrows():
            numbers = [row[f'number_{i}'] for i in range(1, 7)]
            if not validate_lotto_numbers(numbers):
                invalid_count += 1
        
        if invalid_count > 0:
            self.logger.warning(f"유효하지 않은 당첨번호 {invalid_count}개 발견")
        
        self.logger.info(f"데이터 검증 완료: {len(self.processed_data)}개 행 중 {len(self.processed_data) - invalid_count}개 유효")
    
    def get_statistics_summary(self) -> Dict[str, Any]:
        """
        데이터 통계 요약 반환
        
        Returns:
            Dict[str, Any]: 통계 요약 정보
        """
        if self.processed_data is None:
            return {}
        
        df = self.processed_data
        
        # 모든 당첨번호 수집
        all_numbers = []
        for _, row in df.iterrows():
            all_numbers.extend(row['winning_numbers'])
        
        return {
            'total_draws': len(df),
            'latest_round': df['round_number'].max() if not df.empty else None,
            'oldest_round': df['round_number'].min() if not df.empty else None,
            'date_range': {
                'latest': df['draw_date'].iloc[0] if not df.empty and 'draw_date' in df.columns else None,
                'oldest': df['draw_date'].iloc[-1] if not df.empty and 'draw_date' in df.columns else None
            },
            'number_frequency': pd.Series(all_numbers).value_counts().to_dict(),
            'avg_sum': df['number_sum'].mean() if 'number_sum' in df.columns else None,
            'avg_odd_count': df['odd_count'].mean() if 'odd_count' in df.columns else None,
            'avg_even_count': df['even_count'].mean() if 'even_count' in df.columns else None
        }
    
    def save_processed_data(self, output_path: Optional[str] = None):
        """
        처리된 데이터를 CSV로 저장
        
        Args:
            output_path (Optional[str]): 저장할 경로 (기본값: config에서 가져옴)
        """
        if self.processed_data is None:
            raise ValueError("저장할 처리된 데이터가 없습니다.")
        
        if output_path is None:
            output_path = Config.PROCESSED_DATA_PATH
        
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        self.processed_data.to_csv(output_path, index=False, encoding='utf-8')
        self.logger.info(f"처리된 데이터 저장 완료: {output_path}")


def validate_data_integrity(df: pd.DataFrame) -> Dict[str, Any]:
    """
    데이터 무결성 검증 함수
    
    Args:
        df (pd.DataFrame): 검증할 DataFrame
        
    Returns:
        Dict[str, Any]: 검증 결과
    """
    results = {
        'is_valid': True,
        'issues': [],
        'statistics': {}
    }
    
    try:
        # 1. 기본 구조 검증
        if df.empty:
            results['is_valid'] = False
            results['issues'].append("DataFrame이 비어있습니다.")
            return results
        
        # 2. 당첨번호 유효성 검증
        invalid_numbers = 0
        duplicate_numbers = 0
        
        for idx, row in df.iterrows():
            if 'winning_numbers' in row:
                numbers = row['winning_numbers']
                if not validate_lotto_numbers(numbers):
                    invalid_numbers += 1
                
                if len(numbers) != len(set(numbers)):
                    duplicate_numbers += 1
        
        if invalid_numbers > 0:
            results['issues'].append(f"{invalid_numbers}개 행에 유효하지 않은 당첨번호")
        
        if duplicate_numbers > 0:
            results['issues'].append(f"{duplicate_numbers}개 행에 중복 당첨번호")
        
        # 3. 통계 정보
        results['statistics'] = {
            'total_rows': len(df),
            'invalid_numbers': invalid_numbers,
            'duplicate_numbers': duplicate_numbers,
            'valid_rows': len(df) - invalid_numbers - duplicate_numbers
        }
        
        if results['issues']:
            results['is_valid'] = False
        
    except Exception as e:
        results['is_valid'] = False
        results['issues'].append(f"검증 중 오류: {str(e)}")
    
    return results


    # === 모듈 테스트 ===
if __name__ == "__main__":
    # 테스트용 코드 (새로운 CSV 구조용)
    import tempfile
    
    # 테스트 CSV 데이터 생성 (새로운 구조)
    test_csv_content = """year,round,draw_date,1st_winners,1st_prize,2nd_winners,2nd_prize,3rd_winners,3rd_prize,4th_winners,4th_prize,5th_winners,5th_prize,num1,num2,num3,num4,num5,num6,bonus
2025,1187,2025.08.30,11,"2,619,380,012원",79,"60,787,300원","3,147","1,525,961원","152,448","50,000원","2,557,090","5,000원",5,13,26,29,37,40,42
2025,1186,2025.08.23,14,"1,985,676,911원",89,"52,058,946원","3,226","1,436,221원","162,707","50,000원","2,628,810","5,000원",2,8,13,16,23,28,35
2025,1185,2025.08.16,12,"2,388,695,125원",79,"60,473,295원","2,903","1,645,674원","153,798","50,000원","2,566,276","5,000원",6,17,22,28,29,32,38"""
    
    # 임시 파일로 테스트
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False, encoding='utf-8') as f:
        f.write(test_csv_content)
        temp_path = f.name
    
    try:
        print("=== 로또 데이터 로더 테스트 (새로운 CSV 구조) ===")
        
        # 데이터 로더 테스트
        loader = LottoDataLoader(temp_path)
        df = loader.load_and_preprocess()
        
        print(f"로딩된 데이터: {len(df)}행")
        print("\n처리된 데이터 샘플:")
        print(df[['round_number', 'draw_date', 'winning_numbers', 'bonus_number', 'number_sum']].head())
        
        # 통계 요약
        stats = loader.get_statistics_summary()
        print(f"\n통계 요약:")
        print(f"총 추첨 횟수: {stats['total_draws']}")
        print(f"최신 회차: {stats['latest_round']}")
        print(f"가장 오래된 회차: {stats['oldest_round']}")
        print(f"평균 합계: {stats['avg_sum']:.1f}")
        
        # 데이터 무결성 검증
        validation_result = validate_data_integrity(df)
        print(f"\n데이터 무결성 검증:")
        print(f"유효성: {'✅' if validation_result['is_valid'] else '❌'}")
        if validation_result['issues']:
            for issue in validation_result['issues']:
                print(f"  - {issue}")
        
        print("\n✅ 새로운 CSV 구조 테스트 완료")
        
    except Exception as e:
        print(f"테스트 중 오류: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        # 임시 파일 삭제
        import os
        os.unlink(temp_path)
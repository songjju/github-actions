"""
연금복권 720+ 당첨번호 예측 시스템 - 최종 완성 버전
실제 lotto_720.html 파일을 파싱하여 예측
"""

import numpy as np
import pandas as pd
from bs4 import BeautifulSoup
import re
from datetime import datetime, timedelta
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

# 간소화된 예측 시스템 (외부 라이브러리 의존성 최소화)
class Lotto720PredictorFinal:
    """연금복권 720+ 예측 시스템 최종 버전"""
    
    def __init__(self):
        self.data = None
        self.predictions = {}
        np.random.seed(42)  # 재현 가능한 결과
    
    def parse_html_file(self, html_path):
        """HTML 파일 파싱"""
        try:
            # EUC-KR 인코딩으로 시도
            with open(html_path, 'r', encoding='euc-kr') as f:
                html_content = f.read()
        except:
            try:
                # UTF-8 인코딩으로 재시도
                with open(html_path, 'r', encoding='utf-8') as f:
                    html_content = f.read()
            except:
                print("❌ HTML 파일을 읽을 수 없습니다.")
                return False
        
        return self.parse_html_content(html_content)
    
    def parse_html_content(self, html_content):
        """HTML 내용 파싱 - 실제 구조에 맞게 수정"""
        soup = BeautifulSoup(html_content, 'html.parser')
        
        data = []
        all_rows = soup.find_all('tr')
        
        current_year = None
        
        for row in all_rows:
            cells = row.find_all('td')
            
            # 최소 11개 이상의 셀이 있어야 유효한 데이터
            if len(cells) >= 11:
                try:
                    # 연도 추출 (rowspan이 있는 경우)
                    cell_offset = 0
                    
                    # 첫 번째 셀이 연도인지 확인
                    first_cell = cells[0]
                    if first_cell.get('rowspan'):
                        # rowspan이 있으면 연도 셀
                        year_text = first_cell.get_text().strip()
                        if year_text.isdigit() and len(year_text) == 4:
                            current_year = int(year_text)
                            cell_offset = 1
                    elif not current_year:
                        # 연도가 설정되지 않았으면 건너뛰기
                        continue
                    
                    # 데이터 추출 (연도가 별도 셀인 경우 offset 적용)
                    if cell_offset == 0 and len(cells) >= 11:
                        # 연도 셀이 없는 행 (이미 설정된 연도 사용)
                        draw_no_idx = 0
                    else:
                        # 연도 셀이 있는 행
                        draw_no_idx = 1
                    
                    # 회차 번호
                    draw_no_text = cells[draw_no_idx].get_text().strip()
                    if not draw_no_text.isdigit():
                        continue
                    draw_no = int(draw_no_text)
                    
                    # 추첨일
                    draw_date_text = cells[draw_no_idx + 1].get_text().strip()
                    if len(draw_date_text) == 8 and draw_date_text.isdigit():
                        draw_date = pd.to_datetime(draw_date_text, format='%Y%m%d')
                    else:
                        continue
                    
                    # 1등 당첨번호 (조와 번호 포함)
                    first_prize_text = cells[draw_no_idx + 2].get_text().strip()
                    
                    # 조 추출 (N조 형식)
                    jo = 3  # 기본값
                    jo_match = re.search(r'(\d)조', first_prize_text)
                    if jo_match:
                        jo = int(jo_match.group(1))
                    
                    # 6자리 번호 추출
                    numbers = re.findall(r'\d+', first_prize_text)
                    number = None
                    for num in numbers:
                        if len(num) == 6:
                            number = num
                            break
                    
                    if not number:
                        # 6자리 번호를 찾지 못한 경우
                        if numbers and len(numbers[-1]) >= 6:
                            number = numbers[-1][-6:]
                        else:
                            continue
                    
                    # 보너스 번호 (10번째 셀)
                    bonus_idx = draw_no_idx + 9
                    if bonus_idx < len(cells):
                        bonus_text = cells[bonus_idx].get_text().strip()
                        bonus_numbers = re.findall(r'\d+', bonus_text)
                        if bonus_numbers:
                            bonus = bonus_numbers[-1].zfill(6)[-6:]
                        else:
                            bonus = '000000'
                    else:
                        bonus = '000000'
                    
                    # 데이터 저장
                    data.append({
                        'year': current_year,
                        'draw_no': draw_no,
                        'draw_date': draw_date,
                        'jo': jo,
                        'number': number,
                        'bonus': bonus,
                        'digit_0': int(number[0]),
                        'digit_1': int(number[1]),
                        'digit_2': int(number[2]),
                        'digit_3': int(number[3]),
                        'digit_4': int(number[4]),
                        'digit_5': int(number[5])
                    })
                    
                except Exception as e:
                    continue
        
        if data:
            self.data = pd.DataFrame(data)
            self.data = self.data.sort_values('draw_no').reset_index(drop=True)
            
            print(f"✅ 총 {len(self.data)}개의 회차 데이터를 파싱했습니다.")
            print(f"✅ 데이터 범위: {self.data.iloc[0]['draw_no']}회 ~ {self.data.iloc[-1]['draw_no']}회")
            print(f"✅ 최신 회차: {self.data.iloc[-1]['draw_no']}회 ({self.data.iloc[-1]['draw_date'].strftime('%Y-%m-%d')})")
            print(f"✅ 최신 당첨번호: {self.data.iloc[-1]['jo']}조 {self.data.iloc[-1]['number']}")
            return True
        else:
            print("❌ 데이터를 파싱할 수 없습니다.")
            return False
    
    def analyze_patterns(self):
        """패턴 분석"""
        patterns = {}
        
        # 각 자리수별 최빈값
        for i in range(6):
            digit_col = f'digit_{i}'
            freq = self.data[digit_col].value_counts()
            patterns[f'most_common_digit_{i}'] = freq.index[0]
            
            # 최근 30회차 최빈값
            recent_freq = self.data.tail(30)[digit_col].value_counts()
            patterns[f'recent_common_digit_{i}'] = recent_freq.index[0] if len(recent_freq) > 0 else 5
        
        # 조 분석
        jo_freq = self.data['jo'].value_counts()
        patterns['most_common_jo'] = jo_freq.index[0]
        
        # 최근 10회차 조 분석
        recent_jo = self.data.tail(10)['jo'].value_counts()
        patterns['recent_common_jo'] = recent_jo.index[0] if len(recent_jo) > 0 else 3
        
        return patterns
    
    def statistical_prediction(self):
        """통계 기반 예측"""
        prediction = []
        
        for i in range(6):
            # 최근 30회차 가중 빈도
            recent_data = self.data.tail(30)[f'digit_{i}']
            weights = np.linspace(0.5, 1.0, len(recent_data))
            
            digit_weights = {}
            for digit, weight in zip(recent_data, weights):
                if digit not in digit_weights:
                    digit_weights[digit] = 0
                digit_weights[digit] += weight
            
            # 상위 3개 숫자 중 확률적 선택
            sorted_digits = sorted(digit_weights.items(), key=lambda x: x[1], reverse=True)[:3]
            if sorted_digits:
                digits = [d[0] for d in sorted_digits]
                probs = [d[1] for d in sorted_digits]
                probs = np.array(probs) / sum(probs)
                pred_digit = np.random.choice(digits, p=probs)
            else:
                pred_digit = 5
            
            prediction.append(str(pred_digit))
        
        return ''.join(prediction)
    
    def moving_average_prediction(self):
        """이동평균 예측"""
        prediction = []
        
        for i in range(6):
            # 5, 10, 20회차 이동평균
            ma_5 = self.data[f'digit_{i}'].tail(5).mean()
            ma_10 = self.data[f'digit_{i}'].tail(10).mean()
            ma_20 = self.data[f'digit_{i}'].tail(20).mean()
            
            # 가중 평균 (최근 데이터에 더 높은 가중치)
            weighted_avg = 0.5 * ma_5 + 0.3 * ma_10 + 0.2 * ma_20
            
            # 반올림하여 예측
            pred_digit = int(np.clip(np.round(weighted_avg), 0, 9))
            prediction.append(str(pred_digit))
        
        return ''.join(prediction)
    
    def frequency_prediction(self):
        """빈도 기반 예측"""
        prediction = []
        
        for i in range(6):
            # 전체 빈도와 최근 빈도 결합
            all_freq = self.data[f'digit_{i}'].value_counts(normalize=True)
            recent_freq = self.data.tail(50)[f'digit_{i}'].value_counts(normalize=True)
            
            combined_score = {}
            for digit in range(10):
                all_score = all_freq.get(digit, 0.01)
                recent_score = recent_freq.get(digit, 0.01)
                # 최근 데이터에 더 높은 가중치
                combined_score[digit] = 0.3 * all_score + 0.7 * recent_score
            
            # 확률적 선택
            digits = list(combined_score.keys())
            probs = list(combined_score.values())
            probs = np.array(probs) / sum(probs)
            pred_digit = np.random.choice(digits, p=probs)
            prediction.append(str(pred_digit))
        
        return ''.join(prediction)
    
    def pattern_based_prediction(self):
        """패턴 기반 예측"""
        prediction = []
        
        for i in range(6):
            recent = list(self.data.tail(10)[f'digit_{i}'])
            
            # 주기성 탐지 (2-4 주기)
            pattern_found = False
            for period in [2, 3, 4]:
                if len(recent) >= period * 2:
                    # 패턴 매칭 확인
                    matches = 0
                    for j in range(period):
                        if recent[-(period*2) + j] == recent[-period + j]:
                            matches += 1
                    
                    if matches >= period * 0.7:  # 70% 이상 일치
                        pred_digit = recent[len(recent) % period]
                        pattern_found = True
                        break
            
            if not pattern_found:
                # 패턴이 없으면 최빈값 사용
                counter = Counter(recent)
                pred_digit = counter.most_common(1)[0][0] if counter else 5
            
            prediction.append(str(pred_digit))
        
        return ''.join(prediction)
    
    def markov_chain_prediction(self):
        """마르코프 체인 예측"""
        prediction = []
        
        for i in range(6):
            digit_series = self.data[f'digit_{i}'].values
            
            # 전이 행렬 구축
            transition_matrix = np.zeros((10, 10))
            for j in range(len(digit_series)-1):
                current = digit_series[j]
                next_digit = digit_series[j+1]
                transition_matrix[current, next_digit] += 1
            
            # 정규화
            for row in range(10):
                row_sum = transition_matrix[row].sum()
                if row_sum > 0:
                    transition_matrix[row] = transition_matrix[row] / row_sum
                else:
                    transition_matrix[row] = np.ones(10) / 10
            
            # 마지막 상태에서 예측
            last_digit = digit_series[-1]
            probs = transition_matrix[last_digit]
            
            # 확률이 0이 아닌 숫자들 중 선택
            if probs.sum() > 0:
                pred_digit = np.random.choice(10, p=probs)
            else:
                pred_digit = np.random.randint(0, 10)
            
            prediction.append(str(pred_digit))
        
        return ''.join(prediction)
    
    def hybrid_prediction(self):
        """하이브리드 최종 예측"""
        
        if self.data is None or len(self.data) == 0:
            print("❌ 데이터가 없습니다. HTML 파일을 먼저 로드하세요.")
            return None
        
        print("\n" + "="*70)
        print("🎯 연금복권 720+ AI 예측 시스템")
        print("="*70)
        
        # 패턴 분석
        print("\n📊 데이터 패턴 분석 중...")
        patterns = self.analyze_patterns()
        
        # 각 예측 모델 실행
        print("🔮 5개 예측 모델 실행 중...")
        
        predictions = {
            '통계 기반': self.statistical_prediction(),
            '이동평균': self.moving_average_prediction(),
            '빈도 분석': self.frequency_prediction(),
            '패턴 인식': self.pattern_based_prediction(),
            '마르코프 체인': self.markov_chain_prediction()
        }
        
        # 최종 예측 (가중 투표)
        final_prediction = []
        weights_dict = {
            '통계 기반': 0.25,
            '이동평균': 0.20,
            '빈도 분석': 0.25,
            '패턴 인식': 0.15,
            '마르코프 체인': 0.15
        }
        
        for i in range(6):
            digit_scores = {}
            
            for method, pred in predictions.items():
                digit = int(pred[i])
                weight = weights_dict[method]
                
                if digit not in digit_scores:
                    digit_scores[digit] = 0
                digit_scores[digit] += weight
            
            # 가장 높은 점수의 숫자 선택
            best_digit = max(digit_scores.items(), key=lambda x: x[1])[0]
            final_prediction.append(str(best_digit))
        
        final_number = ''.join(final_prediction)
        
        # 조 예측 (최근 30회차 기준)
        jo_freq = Counter(self.data.tail(30)['jo'])
        predicted_jo = jo_freq.most_common(1)[0][0] if jo_freq else 3
        
        # 결과 출력
        self._print_results(predictions, final_number, predicted_jo, patterns)
        
        return {
            'number': final_number,
            'jo': predicted_jo,
            'next_draw': self.data.iloc[-1]['draw_no'] + 1,
            'next_date': (self.data.iloc[-1]['draw_date'] + timedelta(days=7)).strftime('%Y-%m-%d'),
            'individual_predictions': predictions
        }
    
    def _print_results(self, predictions, final_number, predicted_jo, patterns):
        """결과 출력"""
        print("\n" + "="*70)
        print("📋 개별 모델 예측 결과")
        print("="*70)
        
        for method, pred in predictions.items():
            print(f"  {method:15s}: {pred}")
        
        print("\n" + "="*70)
        print("🎊 최종 예측 결과")
        print("="*70)
        
        next_draw = self.data.iloc[-1]['draw_no'] + 1
        next_date = self.data.iloc[-1]['draw_date'] + timedelta(days=7)
        
        print(f"\n  📍 다음 회차: {next_draw}회")
        print(f"  📅 추첨 예정일: {next_date.strftime('%Y-%m-%d')}")
        print(f"\n  🎯 예측 번호: {final_number}")
        print(f"  🎲 예측 조: {predicted_jo}조")
        
        # 최근 5회차 비교
        print("\n" + "="*70)
        print("📈 최근 5회차 당첨번호")
        print("="*70)
        
        for idx in range(min(5, len(self.data))):
            row = self.data.iloc[-(idx+1)]
            print(f"  {row['draw_no']:3d}회: {row['jo']}조 {row['number']} | {row['draw_date'].strftime('%Y-%m-%d')}")
        
        # 자리수별 분석
        print("\n" + "="*70)
        print("🔥 자리수별 최빈값 분석")
        print("="*70)
        
        for i in range(6):
            all_time = patterns.get(f'most_common_digit_{i}', '-')
            recent = patterns.get(f'recent_common_digit_{i}', '-')
            print(f"  {i+1}번째 자리: 전체 최빈값[{all_time}] / 최근 최빈값[{recent}]")
        
        print("\n" + "="*70)
        print("⚠️  중요 안내")
        print("="*70)
        print("  • 이 예측은 과거 데이터의 통계적 분석일 뿐입니다")
        print("  • 연금복권은 완전한 무작위 추첨입니다")
        print("  • 당첨을 보장하지 않으며, 참고용으로만 활용하세요")
        print("  • 책임감 있는 복권 구매를 권장합니다")
        print("="*70 + "\n")
    
    def validate_prediction(self, test_size=20):
        """백테스팅으로 예측 성능 검증"""
        if len(self.data) < test_size + 10:
            print("검증을 위한 충분한 데이터가 없습니다.")
            return
        
        print("\n📊 백테스팅 검증 (최근 20회차)")
        print("-" * 50)
        
        # 원본 데이터 백업
        original_data = self.data.copy()
        
        correct_positions = 0
        total_positions = 0
        
        for test_idx in range(test_size):
            # 테스트할 회차 분리
            train_end_idx = len(original_data) - test_size + test_idx
            self.data = original_data.iloc[:train_end_idx].copy()
            
            # 예측
            pred = self.statistical_prediction()
            actual = original_data.iloc[train_end_idx]['number']
            
            # 자리수별 비교
            for i in range(6):
                if pred[i] == actual[i]:
                    correct_positions += 1
                total_positions += 1
        
        # 원본 데이터 복원
        self.data = original_data
        
        accuracy = (correct_positions / total_positions) * 100
        print(f"  정확도: {accuracy:.2f}% (개별 자리수 일치율)")
        print(f"  평균 일치 자리수: {correct_positions / test_size:.2f}개 / 6개")
        print(f"  예측 성능: {'양호' if accuracy > 10 else '보통'}")
        print("-" * 50)


# 메인 실행 함수
def main():
    """메인 실행 함수"""
    print("🎰 연금복권 720+ AI 예측 시스템 v2.0")
    print("="*70)
    
    # 예측기 생성
    predictor = Lotto720PredictorFinal()
    
    # HTML 파일 경로 설정 (실제 파일 경로로 변경)
    html_file_path = 'lotto_720.html'
    
    # HTML 파일 파싱
    print(f"📂 파일 로드 중: {html_file_path}")
    success = predictor.parse_html_file(html_file_path)
    
    if not success:
        print("\n⚠️  파일을 로드할 수 없습니다. 데모 모드로 실행합니다.")
        
        # 데모 데이터 생성
        demo_data = []
        np.random.seed(42)
        
        for i in range(274):
            number = ''.join([str(np.random.randint(0, 10)) for _ in range(6)])
            demo_data.append({
                'year': 2020 + i // 52,
                'draw_no': i + 1,
                'draw_date': pd.Timestamp('2020-05-07') + pd.Timedelta(days=7*i),
                'jo': np.random.choice([1, 2, 3, 4, 5]),
                'number': number,
                'bonus': ''.join([str(np.random.randint(0, 10)) for _ in range(6)]),
                'digit_0': int(number[0]),
                'digit_1': int(number[1]),
                'digit_2': int(number[2]),
                'digit_3': int(number[3]),
                'digit_4': int(number[4]),
                'digit_5': int(number[5])
            })
        
        predictor.data = pd.DataFrame(demo_data)
        print(f"✅ 데모 데이터 {len(demo_data)}개 생성 완료")
    
    # 예측 실행
    result = predictor.hybrid_prediction()
    
    # 검증 실행
    if len(predictor.data) > 30:
        predictor.validate_prediction()
    
    # 최종 결과 요약
    if result:
        print("\n" + "🎯"*35)
        print(f"\n  💫 275회 예측: {result['jo']}조 {result['number']}")
        print(f"  📅 추첨일: {result['next_date']}")
        print("\n" + "🎯"*35)
    
    return result


if __name__ == "__main__":
    # 프로그램 실행
    result = main()
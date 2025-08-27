"""
Flask 연금복권 720+ 예측 웹 애플리케이션
"""

from flask import Flask, render_template, jsonify, request, session
from flask_cors import CORS
import os
import json
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
from bs4 import BeautifulSoup
import re
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

app = Flask(__name__)
app.secret_key = 'your-secret-key-here-change-this'
CORS(app)

# Lotto720PredictorFinal 클래스 (기존 코드)
class Lotto720PredictorFinal:
    """연금복권 720+ 예측 시스템"""
    
    def __init__(self):
        self.data = None
        self.predictions = {}
        np.random.seed(42)
    
    def parse_html_file(self, html_path):
        """HTML 파일 파싱"""
        try:
            with open(html_path, 'r', encoding='euc-kr') as f:
                html_content = f.read()
        except:
            try:
                with open(html_path, 'r', encoding='utf-8') as f:
                    html_content = f.read()
            except:
                return False
        
        return self.parse_html_content(html_content)
    
    def parse_html_content(self, html_content):
        """HTML 내용 파싱"""
        soup = BeautifulSoup(html_content, 'html.parser')
        
        data = []
        all_rows = soup.find_all('tr')
        
        current_year = None
        
        for row in all_rows:
            cells = row.find_all('td')
            
            if len(cells) >= 11:
                try:
                    cell_offset = 0
                    
                    first_cell = cells[0]
                    if first_cell.get('rowspan'):
                        year_text = first_cell.get_text().strip()
                        if year_text.isdigit() and len(year_text) == 4:
                            current_year = int(year_text)
                            cell_offset = 1
                    elif not current_year:
                        continue
                    
                    if cell_offset == 0 and len(cells) >= 11:
                        draw_no_idx = 0
                    else:
                        draw_no_idx = 1
                    
                    draw_no_text = cells[draw_no_idx].get_text().strip()
                    if not draw_no_text.isdigit():
                        continue
                    draw_no = int(draw_no_text)
                    
                    draw_date_text = cells[draw_no_idx + 1].get_text().strip()
                    if len(draw_date_text) == 8 and draw_date_text.isdigit():
                        draw_date = pd.to_datetime(draw_date_text, format='%Y%m%d')
                    else:
                        continue
                    
                    first_prize_text = cells[draw_no_idx + 2].get_text().strip()
                    
                    jo = 3
                    jo_match = re.search(r'(\d)조', first_prize_text)
                    if jo_match:
                        jo = int(jo_match.group(1))
                    
                    numbers = re.findall(r'\d+', first_prize_text)
                    number = None
                    for num in numbers:
                        if len(num) == 6:
                            number = num
                            break
                    
                    if not number:
                        if numbers and len(numbers[-1]) >= 6:
                            number = numbers[-1][-6:]
                        else:
                            continue
                    
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
            return True
        return False
    
    def statistical_prediction(self):
        """통계 기반 예측"""
        prediction = []
        
        for i in range(6):
            recent_data = self.data.tail(30)[f'digit_{i}']
            weights = np.linspace(0.5, 1.0, len(recent_data))
            
            digit_weights = {}
            for digit, weight in zip(recent_data, weights):
                if digit not in digit_weights:
                    digit_weights[digit] = 0
                digit_weights[digit] += weight
            
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
            ma_5 = self.data[f'digit_{i}'].tail(5).mean()
            ma_10 = self.data[f'digit_{i}'].tail(10).mean()
            ma_20 = self.data[f'digit_{i}'].tail(20).mean()
            
            weighted_avg = 0.5 * ma_5 + 0.3 * ma_10 + 0.2 * ma_20
            pred_digit = int(np.clip(np.round(weighted_avg), 0, 9))
            prediction.append(str(pred_digit))
        
        return ''.join(prediction)
    
    def frequency_prediction(self):
        """빈도 기반 예측"""
        prediction = []
        
        for i in range(6):
            all_freq = self.data[f'digit_{i}'].value_counts(normalize=True)
            recent_freq = self.data.tail(50)[f'digit_{i}'].value_counts(normalize=True)
            
            combined_score = {}
            for digit in range(10):
                all_score = all_freq.get(digit, 0.01)
                recent_score = recent_freq.get(digit, 0.01)
                combined_score[digit] = 0.3 * all_score + 0.7 * recent_score
            
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
            
            pattern_found = False
            for period in [2, 3, 4]:
                if len(recent) >= period * 2:
                    matches = 0
                    for j in range(period):
                        if recent[-(period*2) + j] == recent[-period + j]:
                            matches += 1
                    
                    if matches >= period * 0.7:
                        pred_digit = recent[len(recent) % period]
                        pattern_found = True
                        break
            
            if not pattern_found:
                counter = Counter(recent)
                pred_digit = counter.most_common(1)[0][0] if counter else 5
            
            prediction.append(str(pred_digit))
        
        return ''.join(prediction)
    
    def markov_chain_prediction(self):
        """마르코프 체인 예측"""
        prediction = []
        
        for i in range(6):
            digit_series = self.data[f'digit_{i}'].values
            
            transition_matrix = np.zeros((10, 10))
            for j in range(len(digit_series)-1):
                current = digit_series[j]
                next_digit = digit_series[j+1]
                transition_matrix[current, next_digit] += 1
            
            for row in range(10):
                row_sum = transition_matrix[row].sum()
                if row_sum > 0:
                    transition_matrix[row] = transition_matrix[row] / row_sum
                else:
                    transition_matrix[row] = np.ones(10) / 10
            
            last_digit = digit_series[-1]
            probs = transition_matrix[last_digit]
            
            if probs.sum() > 0:
                pred_digit = np.random.choice(10, p=probs)
            else:
                pred_digit = np.random.randint(0, 10)
            
            prediction.append(str(pred_digit))
        
        return ''.join(prediction)
    
    def hybrid_prediction(self):
        """하이브리드 최종 예측"""
        if self.data is None or len(self.data) == 0:
            return None
        
        predictions = {
            '통계 기반': self.statistical_prediction(),
            '이동평균': self.moving_average_prediction(),
            '빈도 분석': self.frequency_prediction(),
            '패턴 인식': self.pattern_based_prediction(),
            '마르코프 체인': self.markov_chain_prediction()
        }
        
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
            
            best_digit = max(digit_scores.items(), key=lambda x: x[1])[0]
            final_prediction.append(str(best_digit))
        
        final_number = ''.join(final_prediction)
        
        jo_freq = Counter(self.data.tail(30)['jo'])
        predicted_jo = jo_freq.most_common(1)[0][0] if jo_freq else 3
        
        return {
            'number': final_number,
            'jo': predicted_jo,
            'next_draw': int(self.data.iloc[-1]['draw_no'] + 1),
            'next_date': (self.data.iloc[-1]['draw_date'] + timedelta(days=7)).strftime('%Y-%m-%d'),
            'individual_predictions': predictions,
            'confidence': np.random.randint(45, 75)  # 신뢰도 시뮬레이션
        }
    
    def get_recent_draws(self, count=10):
        """최근 당첨번호 조회"""
        recent = []
        for idx in range(min(count, len(self.data))):
            row = self.data.iloc[-(idx+1)]
            recent.append({
                'draw_no': int(row['draw_no']),
                'date': row['draw_date'].strftime('%Y-%m-%d'),
                'jo': int(row['jo']),
                'number': row['number']
            })
        return recent
    
    def get_statistics(self):
        """통계 정보 조회"""
        stats = {
            'digit_frequency': {},
            'jo_frequency': {}
        }
        
        # 각 자리수별 빈도
        for i in range(6):
            freq = self.data[f'digit_{i}'].value_counts().to_dict()
            stats['digit_frequency'][i] = {int(k): int(v) for k, v in freq.items()}
        
        # 조 빈도
        jo_freq = self.data['jo'].value_counts().to_dict()
        stats['jo_frequency'] = {int(k): int(v) for k, v in jo_freq.items()}
        
        return stats


# 전역 예측기 인스턴스
predictor = None

def init_predictor():
    """예측기 초기화"""
    global predictor
    predictor = Lotto720PredictorFinal()
    
    # HTML 파일 경로 (실제 경로로 변경 필요)
    html_file = 'lotto_720.html'
    
    if os.path.exists(html_file):
        success = predictor.parse_html_file(html_file)
        if success:
            print(f"✅ 데이터 로드 성공: {len(predictor.data)}개 회차")
            return True
    
    # 파일이 없으면 데모 데이터 생성
    print("⚠️ HTML 파일이 없어 데모 데이터를 생성합니다.")
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
    return True

# 라우트 정의
@app.route('/')
def index():
    """메인 페이지"""
    return render_template('index.html')

@app.route('/api/predict', methods=['POST'])
def predict():
    """예측 API"""
    if predictor is None:
        return jsonify({'error': '시스템 초기화 중입니다.'}), 500
    
    result = predictor.hybrid_prediction()
    if result:
        # 세션에 저장
        session['last_prediction'] = result
        return jsonify(result)
    else:
        return jsonify({'error': '예측 실패'}), 500

@app.route('/api/recent')
def get_recent():
    """최근 당첨번호 조회"""
    if predictor is None:
        return jsonify({'error': '시스템 초기화 중입니다.'}), 500
    
    count = request.args.get('count', 10, type=int)
    recent = predictor.get_recent_draws(count)
    return jsonify(recent)

@app.route('/api/statistics')
def get_statistics():
    """통계 정보 조회"""
    if predictor is None:
        return jsonify({'error': '시스템 초기화 중입니다.'}), 500
    
    stats = predictor.get_statistics()
    return jsonify(stats)

@app.route('/api/history')
def get_history():
    """예측 히스토리 조회"""
    # 세션에서 마지막 예측 가져오기
    last_prediction = session.get('last_prediction', None)
    if last_prediction:
        return jsonify([last_prediction])
    return jsonify([])

if __name__ == '__main__':
    # 예측기 초기화
    init_predictor()
    
    # Flask 서버 실행
    app.run(debug=True, host='0.0.0.0', port=5000)
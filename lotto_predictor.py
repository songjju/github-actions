import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, RobustScaler, LabelEncoder
from sklearn.metrics import mean_absolute_error, r2_score, accuracy_score
from collections import Counter
import warnings
from datetime import datetime
import pickle
import os
import hashlib
import time

warnings.filterwarnings('ignore')

class LottoPredictor:
    def __init__(self, data, db_manager=None):
        self.data = self._validate_data(data)
        self.db_manager = db_manager
        self.models = {}
        self.scalers = {}
        self.feature_importance = {}
        self.is_trained = False
        self.training_history = []
        
        # 중복 방지를 위한 추가 속성
        self.recent_predictions = []  # 최근 예측 기록
        self.prediction_cache = {}    # 예측 캐시
        self.randomness_seed = int(time.time())  # 시간 기반 시드
        
        # 자릿수별 다양성 추적 (새로 추가)
        self.digit_frequency = {i: {} for i in range(6)}  # 각 자리별 숫자 빈도
        self.digit_history = {i: [] for i in range(6)}    # 각 자리별 최근 사용 기록
        
        print(f"LottoPredictor 초기화: 데이터 {len(self.data)}개")
        
        # 자동으로 모델 학습 시도
        if len(self.data) >= 10:
            try:
                self.train_models()
            except Exception as e:
                print(f"자동 학습 실패, 수동 학습 필요: {e}")

    def _check_digit_diversity(self, number_str):
        """자릿수별 다양성 검사"""
        if len(number_str) != 6:
            return True, "번호 길이 오류"
        
        diversity_issues = []
        
        for pos in range(6):
            digit = int(number_str[pos])
            recent_digits = self.digit_history[pos][-10:]  # 최근 10회 확인
            
            if len(recent_digits) >= 5:  # 최소 5회 이상 기록이 있을 때만 검사
                # 같은 자릿수가 연속으로 5회 이상 나온 경우
                if recent_digits[-5:].count(digit) >= 5:
                    diversity_issues.append(f"{pos+1}번째 자리: {digit}이 과도하게 반복")
                
                # 최근 10회 중 70% 이상이 같은 숫자인 경우
                digit_ratio = recent_digits.count(digit) / len(recent_digits)
                if digit_ratio > 0.7:
                    diversity_issues.append(f"{pos+1}번째 자리: {digit}이 {digit_ratio*100:.0f}% 편중")
        
        if diversity_issues:
            return False, "; ".join(diversity_issues)
        else:
            return True, "자릿수 다양성 양호"
    
    def _update_digit_statistics(self, number_str):
        """자릿수별 통계 업데이트"""
        if len(number_str) != 6:
            return
        
        for pos in range(6):
            digit = int(number_str[pos])
            
            # 빈도 업데이트
            if digit not in self.digit_frequency[pos]:
                self.digit_frequency[pos][digit] = 0
            self.digit_frequency[pos][digit] += 1
            
            # 최근 기록 업데이트
            self.digit_history[pos].append(digit)
            if len(self.digit_history[pos]) > 20:  # 최근 20개만 유지
                self.digit_history[pos] = self.digit_history[pos][-20:]

    def _generate_diverse_digit_number(self, base_number, attempt=0):
        """자릿수 다양성을 고려한 번호 생성"""
        if attempt >= 10:  # 무한 루프 방지
            return str(np.random.randint(100000, 999999)).zfill(6)
        
        base_str = str(base_number).zfill(6)
        new_digits = list(base_str)
        
        # 자릿수별 다양성 검사 및 교체
        for pos in range(6):
            current_digit = int(new_digits[pos])
            recent_digits = self.digit_history[pos][-10:]
            
            if len(recent_digits) >= 3:
                # 현재 숫자가 최근 3회 중 2회 이상 나왔다면 교체
                if recent_digits[-3:].count(current_digit) >= 2:
                    # 가장 적게 사용된 숫자들 찾기
                    digit_counts = {}
                    for d in recent_digits:
                        digit_counts[d] = digit_counts.get(d, 0) + 1
                    
                    # 사용 빈도가 낮은 숫자들 후보
                    min_count = min(digit_counts.values()) if digit_counts else 0
                    candidates = [d for d in range(10) if digit_counts.get(d, 0) <= min_count]
                    
                    # 첫 번째 자리(조)는 1-5만 허용
                    if pos == 0:
                        candidates = [d for d in candidates if 1 <= d <= 5]
                    
                    if candidates:
                        new_digit = np.random.choice(candidates)
                        new_digits[pos] = str(new_digit)
                        print(f"자릿수 다양성 개선: {pos+1}번째 자리 {current_digit} → {new_digit}")
        
        new_number_str = ''.join(new_digits)
        
        # 재귀적으로 다양성 검사
        is_diverse, message = self._check_digit_diversity(new_number_str)
        if not is_diverse and attempt < 5:
            print(f"자릿수 다양성 재시도 {attempt + 1}: {message}")
            return self._generate_diverse_digit_number(int(new_number_str), attempt + 1)
        
        return new_number_str

    def _generate_dynamic_seed(self):
        """동적 시드 생성 - 시간, 데이터, 호출 횟수 기반"""
        current_time = int(time.time() * 1000) % 100000  # 밀리초 단위
        data_hash = hash(str(self.data.tail(5)['number'].tolist())) % 10000
        call_count = len(self.recent_predictions) % 1000
        
        # 복합 시드 생성
        combined = f"{current_time}{data_hash}{call_count}"
        return int(hashlib.md5(combined.encode()).hexdigest()[:8], 16) % 100000

    def _add_prediction_to_history(self, prediction):
        """예측 기록에 추가 (최대 50개 유지)"""
        prediction_key = f"{prediction.get('jo', 0)}_{prediction.get('number', '000000')}"
        self.recent_predictions.append({
            'key': prediction_key,
            'timestamp': datetime.now(),
            'full_prediction': prediction
        })
        
        # 최근 50개만 유지
        if len(self.recent_predictions) > 50:
            self.recent_predictions = self.recent_predictions[-50:]

    def get_digit_diversity_stats(self):
        """자릿수별 다양성 통계 반환"""
        stats = {}
        
        for pos in range(6):
            pos_name = f"position_{pos + 1}"
            recent_digits = self.digit_history[pos][-10:]  # 최근 10개
            
            if not recent_digits:
                stats[pos_name] = {
                    'recent_count': 0,
                    'most_frequent': None,
                    'diversity_score': 100
                }
                continue
            
            # 빈도 분석
            digit_counts = {}
            for d in recent_digits:
                digit_counts[d] = digit_counts.get(d, 0) + 1
            
            most_frequent = max(digit_counts.items(), key=lambda x: x[1]) if digit_counts else (None, 0)
            
            # 다양성 점수 계산 (0-100, 높을수록 다양함)
            unique_digits = len(set(recent_digits))
            max_possible = min(10, len(recent_digits))  # 첫 자리는 1-5만 가능하지만 일반적으로 10으로 계산
            diversity_score = (unique_digits / max_possible) * 100 if max_possible > 0 else 0
            
            stats[pos_name] = {
                'recent_count': len(recent_digits),
                'unique_digits': unique_digits,
                'most_frequent_digit': most_frequent[0],
                'most_frequent_count': most_frequent[1],
                'diversity_score': round(diversity_score, 1),
                'recent_sequence': recent_digits[-5:],  # 최근 5개
                'frequency_distribution': dict(digit_counts)
            }
        
        # 전체 다양성 점수
        all_scores = [stats[f"position_{i+1}"]["diversity_score"] for i in range(6)]
        overall_score = sum(all_scores) / len(all_scores) if all_scores else 0
        
        stats['overall_diversity'] = {
            'score': round(overall_score, 1),
            'status': 'excellent' if overall_score >= 80 else 'good' if overall_score >= 60 else 'needs_improvement'
        }
        
        return stats

    def _is_duplicate_prediction(self, jo, number):
        """중복 예측 검사"""
        if len(self.recent_predictions) == 0:
            return False
            
        prediction_key = f"{jo}_{number}"
        recent_keys = [p['key'] for p in self.recent_predictions[-10:]]  # 최근 10개만 확인
        
        return prediction_key in recent_keys

    def _apply_diversity_boost(self, base_prediction, attempt=0):
        """예측 다양성 향상을 위한 부스팅"""
        if attempt >= 10:  # 무한 루프 방지
            return base_prediction
            
        jo = base_prediction.get('jo', 1)
        number = base_prediction.get('number', '100000')
        
        # 중복 검사
        if self._is_duplicate_prediction(jo, number):
            print(f"중복 감지: {jo}조 {number} (시도: {attempt + 1})")
            return self._generate_alternative_prediction(base_prediction, attempt)
        
        return base_prediction

    def _generate_alternative_prediction(self, base_prediction, attempt):
        """대안 예측 생성"""
        original_jo = base_prediction.get('jo', 1)
        original_number = int(base_prediction.get('number', '100000'))
        
        # 동적 시드로 랜덤성 확보
        dynamic_seed = self._generate_dynamic_seed() + attempt * 1000
        np.random.seed(dynamic_seed)
        
        # 다양한 대안 생성 전략
        strategies = [
            self._shift_number_strategy,
            self._change_jo_strategy, 
            self._pattern_variation_strategy,
            self._random_variation_strategy
        ]
        
        strategy = strategies[attempt % len(strategies)]
        alternative = strategy(original_jo, original_number, attempt)
        
        # 재귀적으로 중복 검사
        if self._is_duplicate_prediction(alternative['jo'], alternative['number']):
            return self._generate_alternative_prediction(base_prediction, attempt + 1)
            
        return alternative

    def _shift_number_strategy(self, jo, number, attempt):
        """번호 시프트 전략"""
        shift_amount = (attempt + 1) * 1000 + np.random.randint(-500, 500)
        new_number = number + shift_amount
        
        # 조에 맞는 범위로 조정
        min_val = jo * 100000
        max_val = (jo + 1) * 100000 - 1
        new_number = max(min_val, min(max_val, new_number))
        
        return {
            'jo': jo,
            'number': str(new_number).zfill(6),
            'method': 'shift',
            'confidence_adjustment': -5  # 신뢰도 약간 감소
        }

    def _change_jo_strategy(self, jo, number, attempt):
        """조 변경 전략"""
        # 최근에 적게 사용된 조 선택
        recent_jos = [p['full_prediction'].get('jo', 1) for p in self.recent_predictions[-20:]]
        jo_counts = Counter(recent_jos)
        
        # 가장 적게 사용된 조들 중 선택
        min_count = min(jo_counts.values()) if jo_counts else 0
        least_used_jos = [j for j in range(1, 6) if jo_counts.get(j, 0) == min_count]
        
        if least_used_jos:
            new_jo = np.random.choice(least_used_jos)
        else:
            new_jo = np.random.randint(1, 6)
        
        # 해당 조 범위에서 번호 생성
        base = new_jo * 100000
        new_number = base + (number % 100000)
        
        return {
            'jo': new_jo,
            'number': str(new_number).zfill(6),
            'method': 'jo_change',
            'confidence_adjustment': -3
        }

    def _pattern_variation_strategy(self, jo, number, attempt):
        """패턴 변형 전략"""
        number_str = str(number).zfill(6)
        digits = [int(d) for d in number_str]
        
        # 몇 개 자릿수를 변경
        num_changes = min(2, attempt + 1)
        change_positions = np.random.choice(6, num_changes, replace=False)
        
        for pos in change_positions:
            # 원래 숫자와 다른 숫자로 변경
            original_digit = digits[pos]
            new_digit = np.random.randint(0, 10)
            while new_digit == original_digit:
                new_digit = np.random.randint(0, 10)
            digits[pos] = new_digit
        
        new_number_str = ''.join(map(str, digits))
        new_jo = int(new_number_str[0])
        
        # 조가 유효하지 않으면 조정
        if new_jo < 1 or new_jo > 5:
            new_jo = jo
            digits[0] = jo
            new_number_str = ''.join(map(str, digits))
        
        return {
            'jo': new_jo,
            'number': new_number_str,
            'method': 'pattern_variation',
            'confidence_adjustment': -7
        }

    def _random_variation_strategy(self, jo, number, attempt):
        """랜덤 변형 전략"""
        # 완전히 새로운 번호 생성
        if attempt < 5:
            # 같은 조에서 새 번호
            base = jo * 100000
            new_number = base + np.random.randint(0, 99999)
        else:
            # 다른 조에서도 허용
            new_jo = np.random.randint(1, 6)
            base = new_jo * 100000
            new_number = base + np.random.randint(0, 99999)
            jo = new_jo
        
        return {
            'jo': jo,
            'number': str(new_number).zfill(6),
            'method': 'random_variation',
            'confidence_adjustment': -10
        }

    def predict_next(self):
        """개선된 예측 함수 - 중복 방지 포함"""
        try:
            if not self.is_trained:
                print("모델이 학습되지 않았습니다. 학습을 시도합니다...")
                self.train_models()
                if not self.is_trained:
                    return self._generate_random_prediction()
            
            print("=== AI 예측 수행 (중복 방지) ===")
            
            # 간단한 모델인 경우
            if 'simple' in self.models:
                return self._generate_pattern_prediction()
            
            # AI 모델 예측
            base_result = self._ml_predict()
            
            # 중복 방지 처리
            main_prediction = base_result.get('ml_prediction', {})
            diversity_result = self._apply_diversity_boost(main_prediction)
            
            # 결과에 다양성 정보 추가
            if 'confidence_adjustment' in diversity_result:
                adjustment = diversity_result.pop('confidence_adjustment')
                base_result['confidence_score'] = max(50, base_result.get('confidence_score', 75) + adjustment)
                base_result['diversity_method'] = diversity_result.pop('method', 'original')
                print(f"다양성 적용: {diversity_result['method']} (신뢰도 조정: {adjustment})")
            
            # 메인 예측 업데이트
            base_result['ml_prediction'].update(diversity_result)
            
            # 예측 기록에 추가
            self._add_prediction_to_history(base_result['ml_prediction'])
            
            # 추가 패턴 예측들도 다양성 적용
            pattern_predictions = base_result.get('pattern_predictions', [])
            diverse_pattern_predictions = []
            
            for pred in pattern_predictions:
                diverse_pred = self._apply_diversity_boost({
                    'jo': pred.get('jo'),
                    'number': pred.get('number')
                })
                
                pred.update({
                    'jo': diverse_pred['jo'],
                    'number': diverse_pred['number']
                })
                diverse_pattern_predictions.append(pred)
            
            base_result['pattern_predictions'] = diverse_pattern_predictions
            
            print(f"=== 중복 방지 예측 완료: {base_result['ml_prediction']['jo']}조 {base_result['ml_prediction']['number']} ===")
            return base_result
            
        except Exception as e:
            print(f"예측 실행 실패: {e}")
            import traceback
            traceback.print_exc()
            return self._generate_random_prediction()

    def _ml_predict(self):
        """ML 모델을 사용한 예측 (기존 코드 유지)"""
        try:
            print("AI 모델로 예측 중...")
            
            # 최신 데이터로 특징 생성
            if len(self.data) == 0:
                raise ValueError("예측을 위한 데이터가 없습니다")
                
            last_features = self.extract_features(self.data.tail(1))
            
            if len(last_features) == 0:
                raise ValueError("특징 추출에 실패했습니다")
            
            predictions = {}
            confidence_scores = {}
            
            # 번호 예측 (앙상블)
            if 'rf' in self.models and 'gb' in self.models and 'number' in self.scalers:
                # 스케일링
                last_features_scaled = self.scalers['number'].transform(last_features)
                
                # 각 모델 예측 - 랜덤성 추가
                dynamic_seed = self._generate_dynamic_seed()
                
                # Random Forest에 랜덤성 추가
                rf_pred = self.models['rf'].predict(last_features_scaled)[0]
                rf_noise = np.random.normal(0, abs(rf_pred) * 0.01)  # 1% 노이즈
                rf_pred += rf_noise
                
                # Gradient Boosting에 랜덤성 추가  
                gb_pred = self.models['gb'].predict(last_features_scaled)[0]
                gb_noise = np.random.normal(0, abs(gb_pred) * 0.01)  # 1% 노이즈
                gb_pred += gb_noise
                
                # 앙상블 예측 (가중 평균)
                ensemble_pred = rf_pred * 0.6 + gb_pred * 0.4
                
                # 유효한 범위로 조정
                ensemble_pred = int(max(100000, min(999999, ensemble_pred)))
                predictions['number'] = str(ensemble_pred).zfill(6)
                
                # 신뢰도 계산 (모델 간 일치도 기반)
                diff_ratio = abs(rf_pred - gb_pred) / max(abs(rf_pred), abs(gb_pred), 1)
                confidence_scores['number'] = max(50, 90 - diff_ratio * 40)
                
                print(f"번호 예측: {predictions['number']} (신뢰도: {confidence_scores['number']:.1f}%)")
            else:
                # 모델이 없으면 패턴 기반
                predictions['number'] = str(np.random.randint(100000, 999999)).zfill(6)
                confidence_scores['number'] = 50.0
            
            # 조 예측
            if 'rf_jo' in self.models:
                last_features_scaled = self.scalers['number'].transform(last_features)
                jo_pred = self.models['rf_jo'].predict(last_features_scaled)[0]
                jo_probs = self.models['rf_jo'].predict_proba(last_features_scaled)[0]
                
                predictions['jo'] = int(jo_pred)
                confidence_scores['jo'] = max(jo_probs) * 100
                
                print(f"조 예측: {predictions['jo']}조 (신뢰도: {confidence_scores['jo']:.1f}%)")
            else:
                # 최근 패턴 기반 조 예측
                recent_jos = self.data.tail(20)['jo'] if len(self.data) >= 20 else self.data['jo']
                jo_counts = recent_jos.value_counts()
                
                # 가장 적게 나온 조 선택 (보정)
                if len(jo_counts) > 0:
                    least_common_jo = jo_counts.idxmin()
                    predictions['jo'] = int(least_common_jo)
                    confidence_scores['jo'] = 65.0
                else:
                    predictions['jo'] = int(predictions['number'][0])
                    confidence_scores['jo'] = 50.0
            
            # 번호와 조 일치성 확인 및 보정
            predicted_jo_from_number = int(predictions['number'][0])
            if predicted_jo_from_number != predictions['jo']:
                # 조에 맞게 번호 조정
                base = predictions['jo'] * 100000
                remainder = int(predictions['number']) % 100000
                adjusted_number = base + remainder
                predictions['number'] = str(adjusted_number).zfill(6)
                print(f"번호-조 일치성 보정: {predictions['number']}")
            
            # 패턴 기반 추가 예측
            pattern_predictions = self._generate_pattern_predictions()
            
            # 전체 신뢰도 계산
            overall_confidence = (confidence_scores.get('number', 50) + confidence_scores.get('jo', 50)) / 2
            
            result = {
                'next_draw_no': int(self.data['draw_no'].max()) + 1,
                'ml_prediction': {
                    'jo': predictions['jo'],
                    'number': predictions['number'],
                    'jo_confidence': round(confidence_scores.get('jo', 50), 2),
                    'number_confidence': round(confidence_scores.get('number', 50), 2)
                },
                'pattern_predictions': pattern_predictions,
                'confidence_score': round(overall_confidence, 2),
                'model_type': 'AI_ML',
                'features_used': len(last_features.columns)
            }
            
            return result
            
        except Exception as e:
            print(f"ML 예측 실패: {e}")
            import traceback
            traceback.print_exc()
            return self._generate_pattern_prediction()

    def get_prediction_diversity_stats(self):
        """예측 다양성 통계"""
        if len(self.recent_predictions) < 5:
            return {"message": "충분한 예측 기록이 없습니다"}
        
        recent_jos = [p['full_prediction'].get('jo', 1) for p in self.recent_predictions[-20:]]
        recent_numbers = [p['full_prediction'].get('number', '100000') for p in self.recent_predictions[-20:]]
        
        jo_diversity = len(set(recent_jos)) / min(len(recent_jos), 5)  # 최대 5개 조
        number_diversity = len(set(recent_numbers)) / len(recent_numbers)
        
        return {
            'jo_diversity': round(jo_diversity, 2),
            'number_diversity': round(number_diversity, 2),
            'recent_predictions_count': len(self.recent_predictions),
            'jo_distribution': dict(Counter(recent_jos)),
            'duplicate_prevention_active': True
        }

    def _validate_data(self, data):
        """데이터 유효성 검사 및 정리 (강화된 버전)"""
        try:
            if data is None or (hasattr(data, 'empty') and data.empty):
                print("데이터가 없어서 샘플 데이터를 생성합니다.")
                return self._create_comprehensive_sample_data()
                
            if isinstance(data, dict):
                df = pd.DataFrame(data)
            elif isinstance(data, pd.DataFrame):
                df = data.copy()
            else:
                df = pd.DataFrame(data)
            
            # 필수 컬럼 확인 및 생성
            if 'number' not in df.columns:
                print("number 컬럼이 없어서 샘플 데이터를 생성합니다.")
                return self._create_comprehensive_sample_data()
            
            # 데이터 타입 정리
            df['number'] = df['number'].astype(str).str.zfill(6)
            df = df[df['number'].str.isdigit()]
            df['number'] = df['number'].astype(int)
            
            # jo 컬럼 생성 또는 검증
            if 'jo' not in df.columns:
                df['jo'] = df['number'].astype(str).str[0].astype(int)
            else:
                df['jo'] = pd.to_numeric(df['jo'], errors='coerce').fillna(1).astype(int)
            
            # 유효한 조 번호만 유지 (1-5)
            df = df[(df['jo'] >= 1) & (df['jo'] <= 5)]
            
            # draw_no 컬럼 생성 또는 검증
            if 'draw_no' not in df.columns:
                df['draw_no'] = range(1, len(df) + 1)
            
            # 날짜 컬럼 처리
            if 'draw_date' in df.columns:
                df['draw_date'] = pd.to_datetime(df['draw_date'], errors='coerce')
            
            # 데이터가 부족하면 샘플 데이터 추가
            if len(df) < 30:
                print(f"데이터가 부족합니다 ({len(df)}개). 샘플 데이터를 추가합니다.")
                sample_df = self._create_comprehensive_sample_data()
                max_draw_no = df['draw_no'].max() if len(df) > 0 else 0
                sample_df['draw_no'] = sample_df['draw_no'] + max_draw_no
                df = pd.concat([df, sample_df], ignore_index=True)
            
            # 중복 제거 및 정렬
            df = df.drop_duplicates(subset=['draw_no']).sort_values('draw_no').reset_index(drop=True)
            
            print(f"데이터 검증 완료: {len(df)}개 (조별 분포: {df['jo'].value_counts().to_dict()})")
            return df
            
        except Exception as e:
            print(f"데이터 검증 실패: {e}")
            import traceback
            traceback.print_exc()
            return self._create_comprehensive_sample_data()
    
    def _create_comprehensive_sample_data(self):
        """포괄적인 샘플 데이터 생성"""
        np.random.seed(42)
        sample_size = 80  # 충분한 학습 데이터
        
        sample_data = []
        base_date = pd.Timestamp('2020-01-01')
        
        for i in range(sample_size):
            # 조별 가중치 적용 (실제와 유사)
            jo_weights = [0.15, 0.25, 0.20, 0.25, 0.15]
            jo = np.random.choice([1, 2, 3, 4, 5], p=jo_weights)
            
            # 조에 맞는 번호 생성
            if jo == 1:
                number = np.random.randint(100000, 199999)
            elif jo == 2:
                number = np.random.randint(200000, 299999)
            elif jo == 3:
                number = np.random.randint(300000, 399999)
            elif jo == 4:
                number = np.random.randint(400000, 499999)
            else:
                number = np.random.randint(500000, 599999)
            
            # 시간적 패턴 추가 (주기성)
            if i % 10 < 3:  # 30% 확률로 특정 패턴
                number = number + (i % 1000)
            
            sample_data.append({
                'draw_no': i + 640,  # 현실적인 회차 번호
                'draw_date': base_date + pd.Timedelta(weeks=i),
                'number': str(number).zfill(6),
                'jo': jo,
                'bonus_number': str(np.random.randint(100000, 999999)).zfill(6)
            })
        
        df = pd.DataFrame(sample_data)
        df['number'] = df['number'].astype(int)
        print(f"포괄적인 샘플 데이터 생성: {len(df)}개")
        return df
        
    def extract_features(self, df):
        """고급 특징 추출"""
        try:
            features = pd.DataFrame(index=df.index)
            
            # 기본 특징들
            df['number_str'] = df['number'].astype(str).str.zfill(6)
            
            # 각 자리수별 특징
            for i in range(6):
                features[f'digit_{i}'] = df['number_str'].str[i].astype(int)
            
            # 조 관련 특징
            features['jo'] = df['jo'].astype(int)
            features['number_int'] = df['number'].astype(int)
            
            # 수학적 특징들
            digit_cols = [f'digit_{i}' for i in range(6)]
            features['digit_sum'] = features[digit_cols].sum(axis=1)
            features['digit_mean'] = features[digit_cols].mean(axis=1)
            features['digit_std'] = features[digit_cols].std(axis=1).fillna(0)
            features['digit_range'] = features[digit_cols].max(axis=1) - features[digit_cols].min(axis=1)
            
            # 홀수/짝수 분석
            features['odd_count'] = sum(features[f'digit_{i}'] % 2 for i in range(6))
            features['even_count'] = 6 - features['odd_count']
            features['odd_ratio'] = features['odd_count'] / 6
            
            # 조별 패턴 분석
            for jo in range(1, 6):
                jo_mask = df['jo'] == jo
                jo_data = df[jo_mask]
                
                if len(jo_data) > 0:
                    # 해당 조의 최근 출현 간격
                    last_idx = jo_data.index[-1] if len(jo_data) > 0 else -1
                    gap = len(df) - 1 - last_idx if last_idx >= 0 else len(df)
                    features[f'jo_{jo}_gap'] = gap
                    
                    # 조별 출현 빈도
                    features[f'jo_{jo}_freq'] = len(jo_data) / len(df) if len(df) > 0 else 0
                    
                    # 조별 평균 번호
                    features[f'jo_{jo}_avg'] = jo_data['number'].mean() if len(jo_data) > 0 else 0
                else:
                    features[f'jo_{jo}_gap'] = len(df)
                    features[f'jo_{jo}_freq'] = 0
                    features[f'jo_{jo}_avg'] = 0
            
            # 시계열 특징 (트렌드)
            if 'draw_no' in df.columns:
                features['draw_no'] = df['draw_no']
                features['draw_no_norm'] = (df['draw_no'] - df['draw_no'].min()) / (df['draw_no'].max() - df['draw_no'].min() + 1)
            
            # 이동 평균 특징 (최근 패턴)
            for window in [5, 10, 20]:
                if len(df) >= window:
                    features[f'ma_{window}_number'] = df['number'].rolling(window=window, min_periods=1).mean()
                    for jo in range(1, 6):
                        jo_recent = df['jo'].rolling(window=window, min_periods=1).apply(lambda x: (x == jo).mean())
                        features[f'ma_{window}_jo_{jo}'] = jo_recent
                else:
                    features[f'ma_{window}_number'] = df['number']
                    for jo in range(1, 6):
                        features[f'ma_{window}_jo_{jo}'] = (df['jo'] == jo).astype(float)
            
            # 연속성 특징
            features['is_consecutive'] = 0
            if len(df) > 1:
                for i in range(1, len(df)):
                    prev_digits = [int(d) for d in str(df.iloc[i-1]['number']).zfill(6)]
                    curr_digits = [int(d) for d in str(df.iloc[i]['number']).zfill(6)]
                    consecutive_count = sum(1 for j in range(5) if abs(curr_digits[j+1] - curr_digits[j]) == 1)
                    features.iloc[i, features.columns.get_loc('is_consecutive')] = consecutive_count
            
            # NaN 값 처리
            features = features.fillna(0)
            
            # 무한값 처리
            features = features.replace([np.inf, -np.inf], 0)
            
            return features
            
        except Exception as e:
            print(f"고급 특징 추출 실패: {e}")
            # 기본 특징만 반환
            basic_features = pd.DataFrame(index=df.index)
            basic_features['jo'] = df['jo'].astype(int)
            basic_features['number_int'] = df['number'].astype(int)
            return basic_features.fillna(0)
        
    def prepare_training_data(self):
        """학습 데이터 준비 (개선된 버전)"""
        try:
            print("학습 데이터 준비 중...")
            
            if len(self.data) < 5:
                raise ValueError("학습을 위한 데이터가 너무 부족합니다")
                
            # 특징 추출
            features = self.extract_features(self.data)
            
            # 타겟 변수 생성
            targets = pd.DataFrame(index=self.data.index)
            
            # 다음 회차 번호 예측을 위한 타겟
            targets['next_number'] = self.data['number'].shift(-1)
            targets['next_jo'] = self.data['jo'].shift(-1)
            
            # 마지막 행 제거 (타겟이 없음)
            features = features[:-1]
            targets = targets[:-1]
            
            # 유효한 데이터만 사용
            valid_idx = targets['next_number'].notna() & targets['next_jo'].notna()
            features = features[valid_idx]
            targets = targets[valid_idx]
            
            if len(features) == 0:
                raise ValueError("유효한 학습 데이터가 없습니다")
            
            print(f"학습 데이터 준비 완료: {len(features)}개")
            print(f"특징 수: {len(features.columns)}개")
            
            return features, targets
            
        except Exception as e:
            print(f"학습 데이터 준비 실패: {e}")
            import traceback
            traceback.print_exc()
            return None, None
    
    def train_models(self):
        """개선된 앙상블 모델 학습"""
        try:
            print("=== AI 모델 학습 시작 ===")
            
            # 데이터 준비
            X, y = self.prepare_training_data()
            
            if X is None or y is None or len(X) < 5:
                print("학습 데이터 부족으로 기본 모델만 생성합니다.")
                self._create_simple_model()
                return
            
            print(f"학습 데이터: {len(X)}개, 특징: {len(X.columns)}개")
            
            # 데이터 분할
            test_size = min(0.3, max(0.1, 1.0 / len(X))) if len(X) > 10 else 0.2
            X_train, X_test, y_train, y_test = train_test_split(
                X, y['next_number'], test_size=test_size, random_state=42, shuffle=False
            )
            
            print(f"훈련 데이터: {len(X_train)}개, 테스트 데이터: {len(X_test)}개")
            
            # 스케일링
            scaler = RobustScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test) if len(X_test) > 0 else X_train_scaled[:1]
            
            # 모델 파라미터 동적 조정
            n_samples = len(X_train)
            n_features = len(X.columns)
            
            # Random Forest 모델
            rf_params = {
                'n_estimators': min(100, max(20, n_samples * 2)),
                'max_depth': min(15, max(5, n_samples // 3)),
                'min_samples_split': min(10, max(2, n_samples // 10)),
                'min_samples_leaf': min(5, max(1, n_samples // 20)),
                'random_state': 42,
                'n_jobs': -1
            }
            
            print(f"Random Forest 학습 중... (파라미터: {rf_params})")
            rf_model = RandomForestRegressor(**rf_params)
            rf_model.fit(X_train_scaled, y_train)
            
            # Gradient Boosting 모델
            gb_params = {
                'n_estimators': min(80, max(10, n_samples)),
                'learning_rate': 0.1,
                'max_depth': min(8, max(3, n_samples // 5)),
                'subsample': 0.8,
                'random_state': 42
            }
            
            print(f"Gradient Boosting 학습 중... (파라미터: {gb_params})")
            gb_model = GradientBoostingRegressor(**gb_params)
            gb_model.fit(X_train_scaled, y_train)
            
            # 조 예측 모델 (분류)
            print("조 예측 모델 학습 중...")
            jo_model = RandomForestClassifier(
                n_estimators=min(50, max(10, n_samples)),
                max_depth=min(10, max(3, n_samples // 5)),
                random_state=42,
                n_jobs=-1
            )
            jo_model.fit(X_train_scaled, y['next_jo'].iloc[:-len(X_test) if len(X_test) > 0 else 0])
            
            # 모델 성능 평가
            if len(X_test) > 0:
                # 회귀 모델 평가
                rf_pred = rf_model.predict(X_test_scaled)
                gb_pred = gb_model.predict(X_test_scaled)
                ensemble_pred = (rf_pred + gb_pred) / 2
                
                mae = mean_absolute_error(y_test, ensemble_pred)
                r2 = r2_score(y_test, ensemble_pred) if len(set(y_test)) > 1 else 0.0
                
                print(f"모델 성능 - MAE: {mae:.2f}, R²: {r2:.4f}")
                
                # 조 예측 성능
                if len(X_test) == len(y['next_jo'].iloc[-len(X_test):]):
                    jo_pred = jo_model.predict(X_test_scaled)
                    jo_actual = y['next_jo'].iloc[-len(X_test):].values
                    jo_accuracy = accuracy_score(jo_actual, jo_pred)
                    print(f"조 예측 정확도: {jo_accuracy:.4f}")
                
                # 교차 검증 (데이터가 충분한 경우)
                if len(X_train) >= 10:
                    cv_scores = cross_val_score(rf_model, X_train_scaled, y_train, cv=min(5, len(X_train)//2), scoring='neg_mean_absolute_error')
                    print(f"교차 검증 MAE: {-cv_scores.mean():.2f} (±{cv_scores.std():.2f})")
            
            # 특징 중요도 분석
            if hasattr(rf_model, 'feature_importances_'):
                feature_importance = dict(zip(X.columns, rf_model.feature_importances_))
                # 상위 10개 특징만 저장
                sorted_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)[:10]
                self.feature_importance = dict(sorted_features)
                print("주요 특징 중요도:")
                for feature, importance in sorted_features[:5]:
                    print(f"  {feature}: {importance:.4f}")
            
            # 모델 저장
            self.models = {
                'rf': rf_model,
                'gb': gb_model,
                'rf_jo': jo_model
            }
            self.scalers = {'number': scaler}
            self.is_trained = True
            
            # 학습 이력 저장
            self.training_history.append({
                'timestamp': datetime.now(),
                'data_size': len(X),
                'features': len(X.columns),
                'mae': mae if 'mae' in locals() else None,
                'r2': r2 if 'r2' in locals() else None
            })
            
            print("=== AI 모델 학습 완료! ===")
            print(f"학습된 모델: {list(self.models.keys())}")
            
        except Exception as e:
            print(f"모델 학습 실패: {e}")
            import traceback
            traceback.print_exc()
            print("기본 모델로 전환합니다.")
            self._create_simple_model()
    
    def _create_simple_model(self):
        """간단한 기본 모델 생성"""
        print("간단한 패턴 기반 모델을 생성합니다.")
        self.models['simple'] = True
        self.is_trained = True
    
    def predict_next(self):
        """다음 회차 예측 (강화된 버전)"""
        try:
            if not self.is_trained:
                print("모델이 학습되지 않았습니다. 학습을 시도합니다...")
                self.train_models()
                if not self.is_trained:
                    return self._generate_random_prediction()
            
            print("=== AI 예측 수행 ===")
            
            # 간단한 모델인 경우
            if 'simple' in self.models:
                return self._generate_pattern_prediction()
            
            # AI 모델 예측
            return self._ml_predict()
            
        except Exception as e:
            print(f"예측 실행 실패: {e}")
            import traceback
            traceback.print_exc()
            return self._generate_random_prediction()
    
    def _ml_predict(self):
        """ML 모델을 사용한 예측"""
        try:
            print("AI 모델로 예측 중...")
            
            # 최신 데이터로 특징 생성
            if len(self.data) == 0:
                raise ValueError("예측을 위한 데이터가 없습니다")
                
            last_features = self.extract_features(self.data.tail(1))
            
            if len(last_features) == 0:
                raise ValueError("특징 추출에 실패했습니다")
            
            predictions = {}
            confidence_scores = {}
            
            # 번호 예측 (앙상블)
            if 'rf' in self.models and 'gb' in self.models and 'number' in self.scalers:
                # 스케일링
                last_features_scaled = self.scalers['number'].transform(last_features)
                
                # 각 모델 예측
                rf_pred = self.models['rf'].predict(last_features_scaled)[0]
                gb_pred = self.models['gb'].predict(last_features_scaled)[0]
                
                # 앙상블 예측 (가중 평균)
                ensemble_pred = rf_pred * 0.6 + gb_pred * 0.4
                
                # 유효한 범위로 조정
                ensemble_pred = int(max(100000, min(999999, ensemble_pred)))
                predictions['number'] = str(ensemble_pred).zfill(6)
                
                # 신뢰도 계산 (모델 간 일치도 기반)
                diff_ratio = abs(rf_pred - gb_pred) / max(abs(rf_pred), abs(gb_pred), 1)
                confidence_scores['number'] = max(50, 90 - diff_ratio * 40)
                
                print(f"번호 예측: {predictions['number']} (신뢰도: {confidence_scores['number']:.1f}%)")
            else:
                # 모델이 없으면 패턴 기반
                predictions['number'] = str(np.random.randint(100000, 999999)).zfill(6)
                confidence_scores['number'] = 50.0
            
            # 조 예측
            if 'rf_jo' in self.models:
                last_features_scaled = self.scalers['number'].transform(last_features)
                jo_pred = self.models['rf_jo'].predict(last_features_scaled)[0]
                jo_probs = self.models['rf_jo'].predict_proba(last_features_scaled)[0]
                
                predictions['jo'] = int(jo_pred)
                confidence_scores['jo'] = max(jo_probs) * 100
                
                print(f"조 예측: {predictions['jo']}조 (신뢰도: {confidence_scores['jo']:.1f}%)")
            else:
                # 최근 패턴 기반 조 예측
                recent_jos = self.data.tail(20)['jo'] if len(self.data) >= 20 else self.data['jo']
                jo_counts = recent_jos.value_counts()
                
                # 가장 적게 나온 조 선택 (보정)
                if len(jo_counts) > 0:
                    least_common_jo = jo_counts.idxmin()
                    predictions['jo'] = int(least_common_jo)
                    confidence_scores['jo'] = 65.0
                else:
                    predictions['jo'] = int(predictions['number'][0])
                    confidence_scores['jo'] = 50.0
            
            # 번호와 조 일치성 확인 및 보정
            predicted_jo_from_number = int(predictions['number'][0])
            if predicted_jo_from_number != predictions['jo']:
                # 조에 맞게 번호 조정
                base = predictions['jo'] * 100000
                remainder = int(predictions['number']) % 100000
                adjusted_number = base + remainder
                predictions['number'] = str(adjusted_number).zfill(6)
                print(f"번호-조 일치성 보정: {predictions['number']}")
            
            # 패턴 기반 추가 예측
            pattern_predictions = self._generate_pattern_predictions()
            
            # 전체 신뢰도 계산
            overall_confidence = (confidence_scores.get('number', 50) + confidence_scores.get('jo', 50)) / 2
            
            result = {
                'next_draw_no': int(self.data['draw_no'].max()) + 1,
                'ml_prediction': {
                    'jo': predictions['jo'],
                    'number': predictions['number'],
                    'jo_confidence': round(confidence_scores.get('jo', 50), 2),
                    'number_confidence': round(confidence_scores.get('number', 50), 2)
                },
                'pattern_predictions': pattern_predictions,
                'confidence_score': round(overall_confidence, 2),
                'model_type': 'AI_ML',
                'features_used': len(last_features.columns)
            }
            
            print(f"=== AI 예측 완료: {predictions['jo']}조 {predictions['number']} (신뢰도: {overall_confidence:.1f}%) ===")
            return result
            
        except Exception as e:
            print(f"ML 예측 실패: {e}")
            import traceback
            traceback.print_exc()
            return self._generate_pattern_prediction()
    
    def _generate_pattern_prediction(self):
        """패턴 기반 예측 (개선된 버전)"""
        try:
            print("패턴 기반 예측 수행 중...")
            
            if len(self.data) == 0:
                return self._generate_random_prediction()
            
            # 최근 20회차 데이터 분석
            recent_data = self.data.tail(20) if len(self.data) >= 20 else self.data
            jo_counts = recent_data['jo'].value_counts()
            
            # 가장 적게 나온 조 선택
            if len(jo_counts) > 0:
                predicted_jo = jo_counts.idxmin()
            else:
                predicted_jo = np.random.randint(1, 6)
            
            # 해당 조의 번호 패턴 분석
            jo_numbers = self.data[self.data['jo'] == predicted_jo]['number']
            
            if len(jo_numbers) > 0:
                # 최근 번호들의 패턴 사용
                recent_jo_numbers = jo_numbers.tail(5)
                
                if len(recent_jo_numbers) >= 2:
                    # 트렌드 계산
                    trend = recent_jo_numbers.iloc[-1] - recent_jo_numbers.iloc[-2]
                    predicted_number = recent_jo_numbers.iloc[-1] + trend
                else:
                    predicted_number = recent_jo_numbers.iloc[-1]
                
                # 범위 조정
                predicted_number = max(predicted_jo * 100000, 
                                     min((predicted_jo + 1) * 100000 - 1, int(predicted_number)))
            else:
                # 해당 조의 기본 범위에서 랜덤
                base = predicted_jo * 100000
                predicted_number = base + np.random.randint(0, 99999)
            
            predicted_number_str = str(int(predicted_number)).zfill(6)
            
            # 추가 패턴 예측들
            pattern_predictions = self._generate_pattern_predictions()
            
            result = {
                'next_draw_no': int(self.data['draw_no'].max()) + 1,
                'ml_prediction': {
                    'jo': int(predicted_jo),
                    'number': predicted_number_str,
                    'jo_confidence': 65.0
                },
                'pattern_predictions': pattern_predictions,
                'confidence_score': 65.0,
                'model_type': 'Pattern_Based'
            }
            
            print(f"패턴 예측 완료: {predicted_jo}조 {predicted_number_str}")
            return result
            
        except Exception as e:
            print(f"패턴 예측 실패: {e}")
            return self._generate_random_prediction()
    
    def _generate_pattern_predictions(self):
        """다양한 패턴 기반 예측들"""
        predictions = []
        
        try:
            if len(self.data) == 0:
                return predictions
                
            # 1. 주기성 기반 예측
            for jo in range(1, 6):
                jo_data = self.data[self.data['jo'] == jo]
                if len(jo_data) > 2:
                    # 출현 간격 계산
                    indices = jo_data.index.tolist()
                    if len(indices) > 1:
                        gaps = [indices[i+1] - indices[i] for i in range(len(indices)-1)]
                        avg_gap = np.mean(gaps)
                        current_gap = len(self.data) - 1 - indices[-1] if indices else 0
                        
                        if current_gap >= avg_gap * 0.8:  # 출현할 시기
                            avg_number = int(jo_data['number'].tail(3).mean())
                            predictions.append({
                                'type': 'cycle',
                                'jo': jo,
                                'number': str(avg_number).zfill(6),
                                'reason': f'{jo}조 주기성 (평균 간격: {avg_gap:.1f})'
                            })
            
            # 2. 트렌드 기반 예측
            if len(self.data) >= 5:
                recent_numbers = self.data.tail(5)['number'].values
                if len(recent_numbers) >= 2:
                    trend = np.polyfit(range(len(recent_numbers)), recent_numbers, 1)[0]
                    next_number = int(recent_numbers[-1] + trend)
                    next_number = max(100000, min(999999, next_number))
                    next_jo = int(str(next_number).zfill(6)[0])
                    
                    predictions.append({
                        'type': 'trend',
                        'jo': next_jo,
                        'number': str(next_number).zfill(6),
                        'reason': f'트렌드 기반 (기울기: {trend:.0f})'
                    })
            
            # 3. 평균 회귀 예측
            for jo in range(1, 6):
                jo_data = self.data[self.data['jo'] == jo]
                if len(jo_data) >= 3:
                    mean_number = int(jo_data['number'].mean())
                    predictions.append({
                        'type': 'mean_reversion',
                        'jo': jo,
                        'number': str(mean_number).zfill(6),
                        'reason': f'{jo}조 평균 회귀'
                    })
            
        except Exception as e:
            print(f"패턴 예측 생성 중 오류: {e}")
        
        return predictions[:5]  # 최대 5개만 반환
    
    def _generate_random_prediction(self):
        """랜덤 예측 생성"""
        random_jo = np.random.randint(1, 6)
        base = random_jo * 100000
        random_number = str(base + np.random.randint(0, 99999)).zfill(6)
        
        result = {
            'next_draw_no': int(self.data['draw_no'].max()) + 1 if len(self.data) > 0 else 721,
            'ml_prediction': {
                'jo': random_jo,
                'number': random_number,
                'jo_confidence': 50.0
            },
            'pattern_predictions': [],
            'confidence_score': 50.0,
            'model_type': 'Random'
        }
        
        print(f"랜덤 예측: {random_jo}조 {random_number}")
        return result
    
    def get_statistics(self):
        """통계 정보 반환"""
        try:
            if len(self.data) == 0:
                return {
                    'total_draws': 0,
                    'jo_distribution': {},
                    'model_status': 'No Data',
                    'last_update': datetime.now().isoformat()
                }
            
            jo_dist = self.data['jo'].value_counts().to_dict()
            
            stats = {
                'total_draws': len(self.data),
                'jo_distribution': jo_dist,
                'model_status': 'Trained' if self.is_trained else 'Not Trained',
                'model_type': list(self.models.keys()) if self.models else [],
                'feature_count': len(self.feature_importance) if self.feature_importance else 0,
                'last_update': datetime.now().isoformat(),
                'data_range': {
                    'min_draw': int(self.data['draw_no'].min()),
                    'max_draw': int(self.data['draw_no'].max())
                } if 'draw_no' in self.data.columns else None
            }
            
            return stats
            
        except Exception as e:
            print(f"통계 생성 실패: {e}")
            return {
                'total_draws': 0,
                'jo_distribution': {},
                'model_status': 'Error',
                'last_update': datetime.now().isoformat()
            }
    
    def save_model(self, filepath='lotto_model.pkl'):
        """모델 저장"""
        try:
            import pickle
            model_data = {
                'models': self.models,
                'scalers': self.scalers,
                'feature_importance': self.feature_importance,
                'is_trained': self.is_trained,
                'training_history': self.training_history
            }
            
            with open(filepath, 'wb') as f:
                pickle.dump(model_data, f)
            
            print(f"모델 저장 완료: {filepath}")
            return True
            
        except Exception as e:
            print(f"모델 저장 실패: {e}")
            return False
    
    def load_model(self, filepath='lotto_model.pkl'):
        """모델 로드"""
        try:
            if not os.path.exists(filepath):
                print(f"모델 파일이 없습니다: {filepath}")
                return False
                
            import pickle
            with open(filepath, 'rb') as f:
                model_data = pickle.load(f)
            
            self.models = model_data.get('models', {})
            self.scalers = model_data.get('scalers', {})
            self.feature_importance = model_data.get('feature_importance', {})
            self.is_trained = model_data.get('is_trained', False)
            self.training_history = model_data.get('training_history', [])
            
            print(f"모델 로드 완료: {filepath}")
            return True
            
        except Exception as e:
            print(f"모델 로드 실패: {e}")
            return False
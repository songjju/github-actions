import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.metrics import mean_absolute_error, r2_score
from collections import Counter
import warnings
from datetime import datetime
warnings.filterwarnings('ignore')

class LottoPredictor:
    def __init__(self, data, db_manager=None):
        self.data = self._validate_data(data)
        self.db_manager = db_manager
        self.models = {}
        self.scalers = {}
        self.feature_importance = {}
        self.is_trained = False
        
        print(f"LottoPredictor 초기화: 데이터 {len(self.data)}개")
        
    def _validate_data(self, data):
        """데이터 유효성 검사 및 정리"""
        try:
            if data is None:
                return self._create_sample_data()
                
            if isinstance(data, dict):
                df = pd.DataFrame(data)
            elif isinstance(data, pd.DataFrame):
                df = data.copy()
            else:
                df = pd.DataFrame(data)
            
            # 필수 컬럼 확인
            if 'number' not in df.columns:
                return self._create_sample_data()
            
            # number 컬럼을 문자열로 변환 후 숫자 확인
            df['number'] = df['number'].astype(str)
            df = df[df['number'].str.isdigit()]
            df['number'] = df['number'].astype(int)
            
            # jo 컬럼이 없으면 생성
            if 'jo' not in df.columns:
                df['jo'] = df['number'].astype(str).str[0].astype(int)
            
            # draw_no 컬럼이 없으면 생성
            if 'draw_no' not in df.columns:
                df['draw_no'] = range(1, len(df) + 1)
            
            if len(df) < 10:
                print("데이터가 부족하여 샘플 데이터를 추가합니다.")
                sample_df = self._create_sample_data()
                df = pd.concat([df, sample_df], ignore_index=True)
            
            return df.reset_index(drop=True)
            
        except Exception as e:
            print(f"데이터 검증 실패: {e}")
            return self._create_sample_data()
    
    def _create_sample_data(self):
        """샘플 데이터 생성"""
        np.random.seed(42)
        sample_size = 50
        
        sample_data = []
        for i in range(sample_size):
            number = np.random.randint(100000, 600000)
            sample_data.append({
                'draw_no': i + 1,
                'number': number,
                'jo': int(str(number)[0])
            })
        
        df = pd.DataFrame(sample_data)
        print(f"샘플 데이터 생성: {len(df)}개")
        return df
        
    def extract_features(self, df):
        """특징 추출"""
        try:
            features = pd.DataFrame(index=df.index)
            
            # 기본 특징
            df['number_str'] = df['number'].astype(str).str.zfill(6)
            for i in range(6):
                features[f'digit_{i}'] = df['number_str'].str[i].astype(int)
            
            features['jo'] = df['jo'].astype(int)
            
            # 조별 특징
            for jo in range(1, 6):
                jo_data = df[df['jo'] == jo]
                if len(jo_data) > 0:
                    # 해당 조의 최근 출현 위치
                    last_occurrence = jo_data.index[-1] if len(jo_data) > 0 else -1
                    gap = len(df) - 1 - last_occurrence if last_occurrence >= 0 else len(df)
                    features[f'jo_{jo}_gap'] = gap
                    
                    # 조별 빈도
                    features[f'jo_{jo}_freq'] = len(jo_data) / len(df)
                else:
                    features[f'jo_{jo}_gap'] = len(df)
                    features[f'jo_{jo}_freq'] = 0
            
            # 최근 패턴
            for n in [5, 10]:
                if len(df) >= n:
                    recent = df.tail(n)
                    recent_jos = recent['jo'].value_counts()
                    for jo in range(1, 6):
                        features[f'recent_{n}_jo_{jo}'] = recent_jos.get(jo, 0) / n
                else:
                    for jo in range(1, 6):
                        features[f'recent_{n}_jo_{jo}'] = 0
            
            # 수치적 특징
            features['number_int'] = df['number'].astype(int)
            features['number_sum'] = features[[f'digit_{i}' for i in range(6)]].sum(axis=1)
            features['number_mean'] = features[[f'digit_{i}' for i in range(6)]].mean(axis=1)
            
            # 패턴 특징
            features['odd_count'] = sum(features[f'digit_{i}'] % 2 for i in range(6))
            features['even_count'] = 6 - features['odd_count']
            
            # NaN 처리
            features = features.fillna(0)
            
            return features
            
        except Exception as e:
            print(f"특징 추출 실패: {e}")
            # 기본 특징만 반환
            basic_features = pd.DataFrame(index=df.index)
            basic_features['jo'] = df['jo'].astype(int)
            basic_features['number_int'] = df['number'].astype(int)
            return basic_features.fillna(0)
        
    def prepare_training_data(self):
        """학습 데이터 준비"""
        try:
            if len(self.data) < 5:
                print("Warning: 학습을 위한 데이터가 매우 부족합니다")
                
            features = self.extract_features(self.data)
            
            # 타겟 변수
            targets = pd.DataFrame(index=self.data.index)
            targets['next_number'] = self.data['number'].shift(-1).astype(float)
            targets['next_jo'] = self.data['jo'].shift(-1)
            
            # 마지막 행 제거 (타겟이 없음)
            features = features[:-1]
            targets = targets[:-1]
            
            # NaN 제거
            valid_idx = targets['next_number'].notna()
            features = features[valid_idx]
            targets = targets[valid_idx]
            
            print(f"학습 데이터 준비 완료: {len(features)}개")
            return features, targets
            
        except Exception as e:
            print(f"학습 데이터 준비 실패: {e}")
            return None, None
    
    def train_models(self):
        """앙상블 모델 학습"""
        try:
            print("모델 학습을 시작합니다...")
            
            X, y = self.prepare_training_data()
            
            if X is None or y is None or len(X) < 3:
                print("학습 데이터 부족으로 간단한 모델만 생성합니다.")
                self._create_simple_model()
                return
            
            # 훈련/테스트 분할
            if len(X) > 10:
                test_size = 0.2
            else:
                test_size = 1  # 테스트 데이터 1개
                
            X_train, X_test, y_train, y_test = train_test_split(
                X, y['next_number'], test_size=test_size, random_state=42
            )
            
            # 스케일링
            scaler_number = RobustScaler()
            X_train_scaled = scaler_number.fit_transform(X_train)
            X_test_scaled = scaler_number.transform(X_test)
            
            # 모델 생성 (데이터 크기에 맞게 조정)
            n_samples = len(X_train)
            
            rf_model = RandomForestRegressor(
                n_estimators=min(50, max(10, n_samples)),
                max_depth=min(8, max(3, n_samples // 3)),
                min_samples_split=min(5, max(2, n_samples // 5)),
                min_samples_leaf=1,
                random_state=42,
                n_jobs=1
            )
            
            gb_model = GradientBoostingRegressor(
                n_estimators=min(50, max(10, n_samples)),
                learning_rate=0.1,
                max_depth=min(5, max(2, n_samples // 5)),
                subsample=0.8,
                random_state=42
            )
            
            # 학습
            print(f"Random Forest 학습 중... (데이터: {len(X_train)}개)")
            rf_model.fit(X_train_scaled, y_train)
            
            print(f"Gradient Boosting 학습 중... (데이터: {len(X_train)}개)")
            gb_model.fit(X_train_scaled, y_train)
            
            # 예측 및 평가
            if len(X_test) > 0:
                rf_pred = rf_model.predict(X_test_scaled)
                gb_pred = gb_model.predict(X_test_scaled)
                ensemble_pred = (rf_pred + gb_pred) / 2
                
                test_mae = mean_absolute_error(y_test, ensemble_pred)
                
                if np.var(y_test) > 0:
                    test_r2 = r2_score(y_test, ensemble_pred)
                else:
                    test_r2 = 0.0
                    
                print(f"테스트 MAE: {test_mae:.2f}")
                print(f"테스트 R2: {test_r2:.4f}")
            
            # 특징 중요도
            if hasattr(rf_model, 'feature_importances_'):
                self.feature_importance = dict(zip(X.columns, rf_model.feature_importances_))
            
            # 모델 저장
            self.models['rf'] = rf_model
            self.models['gb'] = gb_model
            self.scalers['number'] = scaler_number
            
            # 조 예측 모델
            self._train_jo_model(X, y)
            
            self.is_trained = True
            print("모델 학습 완료!")
            
        except Exception as e:
            print(f"모델 학습 실패: {e}")
            import traceback
            traceback.print_exc()
            self._create_simple_model()
    
    def _train_jo_model(self, X, y):
        """조 예측 모델 학습"""
        try:
            X_jo = X.copy()
            y_jo = y['next_jo'].copy()
            
            # 유효한 조 값만 사용
            valid_jo_mask = y_jo.between(1, 5)
            X_jo = X_jo[valid_jo_mask]
            y_jo = y_jo[valid_jo_mask]
            
            if len(X_jo) < 3:
                print("조 예측 모델을 위한 데이터 부족")
                return
            
            unique_jos = y_jo.unique()
            print(f"조 예측 모델 학습: {len(unique_jos)}개 클래스")
            
            rf_jo = RandomForestClassifier(
                n_estimators=min(30, len(X_jo)),
                max_depth=min(5, len(unique_jos) + 2),
                random_state=42,
                n_jobs=1
            )
            
            rf_jo.fit(X_jo, y_jo)
            
            # 조 예측 정확도 확인
            if len(X_jo) > 3:
                jo_accuracy = rf_jo.score(X_jo, y_jo)
                print(f"조 예측 정확도: {jo_accuracy:.4f}")
            
            self.models['rf_jo'] = rf_jo
            
        except Exception as e:
            print(f"조 예측 모델 학습 실패: {e}")
    
    def _create_simple_model(self):
        """간단한 기본 모델 생성"""
        print("간단한 기본 모델을 생성합니다.")
        
        # 최소한의 모델 생성
        self.models['simple'] = True
        self.is_trained = True
    
    def predict_next(self):
        """다음 회차 예측"""
        try:
            if not self.is_trained:
                print("No models trained, returning random prediction")
                return self._generate_random_prediction()
            
            # 간단한 모델인 경우
            if 'simple' in self.models:
                return self._generate_pattern_prediction()
            
            # ML 모델 예측
            return self._ml_predict()
            
        except Exception as e:
            print(f"예측 실행 실패: {e}")
            return self._generate_random_prediction()
    
    def _ml_predict(self):
        """ML 모델을 사용한 예측"""
        try:
            # 최근 데이터로 특징 생성
            last_features = self.extract_features(self.data.tail(1))
            
            predictions = {}
            
            # 번호 예측
            if 'rf' in self.models and 'gb' in self.models and 'number' in self.scalers:
                last_features_scaled = self.scalers['number'].transform(last_features)
                
                rf_pred = self.models['rf'].predict(last_features_scaled)[0]
                gb_pred = self.models['gb'].predict(last_features_scaled)[0]
                ensemble_pred = (rf_pred + gb_pred) / 2
                
                # 유효한 범위로 클리핑
                ensemble_pred = int(max(100000, min(999999, ensemble_pred)))
                predictions['number'] = str(ensemble_pred).zfill(6)
            else:
                predictions['number'] = str(np.random.randint(100000, 999999)).zfill(6)
            
            # 조 예측
            if 'rf_jo' in self.models:
                jo_pred = self.models['rf_jo'].predict(last_features)[0]
                jo_probs = self.models['rf_jo'].predict_proba(last_features)[0]
                jo_confidence = max(jo_probs) * 100
                predictions['jo'] = int(jo_pred)
            else:
                # 가장 빈번한 조 또는 랜덤
                jo_counts = self.data['jo'].value_counts()
                predictions['jo'] = int(jo_counts.index[0]) if len(jo_counts) > 0 else np.random.randint(1, 6)
                jo_confidence = 60.0
            
            # 패턴 기반 예측
            pattern_predictions = self._pattern_based_predictions()
            
            # 결과 구성
            result = {
                'next_draw_no': int(self.data['draw_no'].max()) + 1,
                'ml_prediction': {
                    'jo': predictions['jo'],
                    'number': predictions['number'],
                    'jo_confidence': round(jo_confidence, 2)
                },
                'pattern_predictions': pattern_predictions,
                'confidence_score': self._calculate_confidence()
            }
            
            print(f"ML 예측 완료: {predictions['jo']}조 {predictions['number']}")
            return result
            
        except Exception as e:
            print(f"ML 예측 실패: {e}")
            return self._generate_random_prediction()
    
    def _generate_pattern_prediction(self):
        """패턴 기반 예측"""
        try:
            # 최근 조 분포 분석
            recent_jos = self.data.tail(20)['jo'] if len(self.data) >= 20 else self.data['jo']
            jo_counts = recent_jos.value_counts()
            
            # 가장 적게 나온 조 선택
            if len(jo_counts) > 0:
                least_common_jo = jo_counts.idxmin()
            else:
                least_common_jo = np.random.randint(1, 6)
            
            # 해당 조의 최근 번호들의 평균 사용
            jo_numbers = self.data[self.data['jo'] == least_common_jo]['number']
            if len(jo_numbers) > 0:
                avg_number = int(jo_numbers.tail(5).mean())
                predicted_number = str(max(100000, min(999999, avg_number))).zfill(6)
            else:
                base = least_common_jo * 100000
                predicted_number = str(base + np.random.randint(0, 99999)).zfill(6)
            
            pattern_predictions = self._pattern_based_predictions()
            
            result = {
                'next_draw_no': int(self.data['draw_no'].max()) + 1,
                'ml_prediction': {
                    'jo': int(least_common_jo),
                    'number': predicted_number,
                    'jo_confidence': 65.0
                },
                'pattern_predictions': pattern_predictions,
                'confidence_score': 65.0
            }
            
            print(f"패턴 예측 완료: {least_common_jo}조 {predicted_number}")
            return result
            
        except Exception as e:
            print(f"패턴 예측 실패: {e}")
            return self._generate_random_prediction()
    
    def _generate_random_prediction(self):
        """랜덤 예측 생성"""
        random_jo = np.random.randint(1, 6)
        base = random_jo * 100000
        random_number = str(base + np.random.randint(0, 99999)).zfill(6)
        
        result = {
            'next_draw_no': int(self.data['draw_no'].max()) + 1 if len(self.data) > 0 else 1,
            'ml_prediction': {
                'jo': random_jo,
                'number': random_number,
                'jo_confidence': 50.0
            },
            'pattern_predictions': [],
            'confidence_score': 50.0
        }
        
        print(f"랜덤 예측: {random_jo}조 {random_number}")
        return result
    
    def _pattern_based_predictions(self):
        """패턴 기반 추가 예측"""
        try:
            predictions = []
            
            # 조별 주기성 분석
            for target_jo in range(1, 6):
                jo_data = self.data[self.data['jo'] == target_jo]
                if len(jo_data) > 2:
                    # 최근 출현 간격
                    indices = jo_data.index.tolist()
                    current_gap = len(self.data) - 1 - indices[-1] if indices else 0
                    
                    # 평균 간격
                    if len(indices) > 1:
                        gaps = [indices[i+1] - indices[i] for i in range(len(indices)-1)]
                        avg_gap = np.mean(gaps)
                    else:
                        avg_gap = len(self.data) / 5  # 기본값
                    
                    # 출현 가능성
                    if current_gap >= avg_gap * 0.7:
                        recent_numbers = jo_data.tail(3)['number'].astype(int).values
                        if len(recent_numbers) > 0:
                            predicted_number = str(int(np.median(recent_numbers))).zfill(6)
                            
                            predictions.append({
                                'jo': target_jo,
                                'number': predicted_number,
                                'confidence': min(85, 45 + current_gap / avg_gap * 15),
                                'reason': f'주기성 분석 (평균 {avg_gap:.1f}회, 현재 {current_gap}회)'
                            })
            
            # 상위 3개만 반환
            return sorted(predictions, key=lambda x: x['confidence'], reverse=True)[:3]
            
        except Exception as e:
            print(f"패턴 분석 실패: {e}")
            return []
    
    def _calculate_confidence(self):
        """예측 신뢰도 계산"""
        try:
            factors = {
                'data_size': min(len(self.data) / 100, 1.0),
                'model_performance': 0.7 if self.is_trained else 0.5,
                'feature_quality': 0.65
            }
            
            confidence = np.mean(list(factors.values())) * 100
            return round(max(50.0, min(95.0, confidence)), 2)
            
        except Exception:
            return 65.0
    
    def get_statistics(self):
        """통계 정보"""
        try:
            if len(self.data) == 0:
                return self._get_default_stats()
            
            # 기본 통계
            jo_distribution = self.data['jo'].value_counts().to_dict()
            recent_10_jos = self.data.tail(10)['jo'].tolist() if len(self.data) >= 10 else self.data['jo'].tolist()
            
            numbers = self.data['number'].astype(int)
            avg_number = int(numbers.mean())
            std_number = int(numbers.std()) if len(numbers) > 1 else 0
            median_number = int(numbers.median())
            
            # 패턴 통계 계산
            pattern_stats = self._calculate_pattern_stats()
            
            # 조별 통계
            jo_stats = {}
            for jo in range(1, 6):
                jo_data = self.data[self.data['jo'] == jo]
                if len(jo_data) > 0:
                    last_idx = jo_data.index[-1]
                    last_appearance = len(self.data) - 1 - last_idx
                    
                    jo_stats[f'jo_{jo}'] = {
                        'count': len(jo_data),
                        'percentage': round(len(jo_data) / len(self.data) * 100, 2),
                        'avg_number': int(jo_data['number'].astype(int).mean()),
                        'last_appearance': last_appearance
                    }
                else:
                    jo_stats[f'jo_{jo}'] = {
                        'count': 0,
                        'percentage': 0.0,
                        'avg_number': jo * 100000 + 50000,
                        'last_appearance': len(self.data)
                    }
            
            stats = {
                'total_draws': len(self.data),
                'jo_distribution': jo_distribution,
                'recent_10_jos': recent_10_jos,
                'avg_number': avg_number,
                'std_number': std_number,
                'median_number': median_number,
                'pattern_stats': pattern_stats,
                'jo_stats': jo_stats
            }
            
            return stats
            
        except Exception as e:
            print(f"통계 생성 실패: {e}")
            return self._get_default_stats()
    
    def _calculate_pattern_stats(self):
        """패턴 통계 계산"""
        try:
            pattern_stats = {
                'avg_odd_ratio': 0.5,
                'avg_high_ratio': 0.5,
                'avg_consecutive': 0.2,
                'avg_repeat': 0.1
            }
            
            if len(self.data) > 0:
                # 홀수 비율 계산
                odd_ratios = []
                for _, row in self.data.iterrows():
                    number_str = str(row['number']).zfill(6)
                    odd_count = sum(1 for d in number_str if int(d) % 2 == 1)
                    odd_ratios.append(odd_count / 6)
                
                pattern_stats['avg_odd_ratio'] = round(np.mean(odd_ratios), 3)
                
                # 높은 숫자 비율 (5 이상)
                high_ratios = []
                for _, row in self.data.iterrows():
                    number_str = str(row['number']).zfill(6)
                    high_count = sum(1 for d in number_str if int(d) >= 5)
                    high_ratios.append(high_count / 6)
                
                pattern_stats['avg_high_ratio'] = round(np.mean(high_ratios), 3)
            
            return pattern_stats
            
        except Exception as e:
            print(f"패턴 통계 계산 실패: {e}")
            return {
                'avg_odd_ratio': 0.5,
                'avg_high_ratio': 0.5,
                'avg_consecutive': 0.2,
                'avg_repeat': 0.1
            }
    
    def _get_default_stats(self):
        """기본 통계 정보"""
        return {
            'total_draws': 0,
            'jo_distribution': {1: 0, 2: 0, 3: 0, 4: 0, 5: 0},
            'recent_10_jos': [],
            'avg_number': 300000,
            'std_number': 100000,
            'median_number': 300000,
            'pattern_stats': {
                'avg_odd_ratio': 0.5,
                'avg_high_ratio': 0.5,
                'avg_consecutive': 0.2,
                'avg_repeat': 0.1
            },
            'jo_stats': {
                f'jo_{jo}': {
                    'count': 0,
                    'percentage': 0.0,
                    'avg_number': jo * 100000 + 50000,
                    'last_appearance': 0
                } for jo in range(1, 6)
            }
        }
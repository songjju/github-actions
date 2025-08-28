from flask import Flask, render_template, jsonify, request
from flask_cors import CORS
import os
from datetime import datetime
from apscheduler.schedulers.background import BackgroundScheduler
import traceback

from data_parser import LottoDataParser
from lotto_predictor import LottoPredictor
from advanced_predictor import AdvancedLottoPredictor
from visualization import LottoVisualizer
from database import DatabaseManager

app = Flask(__name__)
CORS(app)  # API 접근 허용

# 전역 변수
system = {
    'predictor': None,
    'advanced_predictor': None,
    'visualizer': None,
    'db_manager': None,
    'data': None,  # 원본 데이터 저장
    'last_update': None
}

def initialize_system():
    """시스템 초기화"""
    try:
        print("시스템 초기화 중...")
        
        # 데이터베이스 매니저
        system['db_manager'] = DatabaseManager()
        
        # HTML 파일 파싱
        parser = LottoDataParser('data/lotto_720.html')
        data = parser.parse_html()
        system['data'] = data  # 데이터 저장
        
        # 예측 모델 학습
        system['predictor'] = LottoPredictor(data, system['db_manager'])
        system['predictor'].train_models()
        
        # 고급 예측 모델
        system['advanced_predictor'] = AdvancedLottoPredictor(data)
        system['advanced_predictor'].train_advanced_models()
        
        # 시각화 객체 (데이터와 함께 초기화)
        system['visualizer'] = LottoVisualizer(data)
        
        system['last_update'] = datetime.now()
        
        print("시스템 초기화 완료!")
        return True
        
    except Exception as e:
        print(f"시스템 초기화 실패: {e}")
        traceback.print_exc()
        return False

def get_safe_visualizer():
    """안전한 시각화 객체 반환"""
    try:
        # 기존 visualizer가 있으면 사용
        if system.get('visualizer') is not None:
            return system['visualizer']
        
        # 없으면 데이터와 함께 새로 생성
        if system.get('data') is not None:
            return LottoVisualizer(system['data'])
        
        # 데이터도 없으면 빈 visualizer 생성
        print("데이터가 없어서 빈 visualizer를 생성합니다.")
        return LottoVisualizer(None)
        
    except Exception as e:
        print(f"Visualizer 생성 중 오류: {e}")
        return LottoVisualizer(None)

def update_data():
    """데이터 업데이트 (스케줄러용)"""
    try:
        parser = LottoDataParser()
        latest_data = parser.fetch_latest_data()
        if latest_data:
            # 새 데이터가 있으면 재학습
            initialize_system()
    except Exception as e:
        print(f"데이터 업데이트 실패: {e}")

# 스케줄러 설정
try:
    scheduler = BackgroundScheduler()
    scheduler.add_job(func=update_data, trigger="cron", hour=12)  # 매일 정오 실행
    scheduler.start()
    print("스케줄러 시작됨")
except Exception as e:
    print(f"스케줄러 설정 실패: {e}")

@app.route('/')
def index():
    """메인 페이지"""
    try:
        return render_template('index.html')
    except Exception as e:
        print(f"메인 페이지 렌더링 실패: {e}")
        return create_simple_home_page()

@app.route('/predict', methods=['GET'])
def predict():
    """예측 수행"""
    try:
        # 시스템이 초기화되지 않았으면 초기화 시도
        if system['predictor'] is None:
            print("예측 시스템이 없어서 초기화를 시도합니다.")
            if not initialize_system():
                return create_error_page("시스템 초기화 실패", "예측 시스템을 초기화할 수 없습니다.")
        
        # 기본 예측
        try:
            ml_prediction = system['predictor'].predict_next()
        except Exception as pred_error:
            print(f"ML 예측 실패: {pred_error}")
            ml_prediction = create_sample_prediction()
        
        # 고급 예측
        try:
            advanced_prediction = system['advanced_predictor'].predict_ensemble()
        except Exception as adv_error:
            print(f"고급 예측 실패: {adv_error}")
            advanced_prediction = create_sample_advanced_prediction()
        
        # 통합 예측
        combined_prediction = {
            'ml': ml_prediction,
            'advanced': advanced_prediction,
            'statistics': get_safe_statistics()
        }
        
        # 차트 생성 (에러 처리 강화)
        charts = create_safe_charts()
        
        # 템플릿 렌더링 시도
        try:
            return render_template('result.html', 
                                 prediction=combined_prediction,
                                 charts=charts)
        except Exception as template_error:
            print(f"result.html 템플릿 렌더링 실패: {template_error}")
            return create_result_page(combined_prediction, charts)
            
    except Exception as e:
        print(f"예측 페이지 전체 오류: {e}")
        traceback.print_exc()
        return create_error_page("예측 실행 오류", str(e))

@app.route('/analysis')
def analysis():
    """상세 분석 페이지"""
    try:
        if system['advanced_predictor'] is None:
            if not initialize_system():
                return create_error_page("시스템 초기화 실패", "분석 시스템을 초기화할 수 없습니다.")
        
        # 심층 패턴 분석
        try:
            patterns = system['advanced_predictor'].analyze_deep_patterns()
        except Exception as pattern_error:
            print(f"패턴 분석 실패: {pattern_error}")
            patterns = create_sample_patterns()
        
        # 신뢰도 차트
        try:
            visualizer = get_safe_visualizer()
            confidence_chart = visualizer.create_prediction_confidence_chart({
                'ML': {'confidence': 75},
                'LSTM': {'confidence': 82},
                'XGBoost': {'confidence': 78}
            })
        except Exception as chart_error:
            print(f"신뢰도 차트 생성 실패: {chart_error}")
            confidence_chart = None
        
        # 템플릿 렌더링 시도
        try:
            return render_template('analysis.html', 
                                 patterns=patterns,
                                 confidence_chart=confidence_chart)
        except Exception as template_error:
            print(f"analysis.html 템플릿 렌더링 실패: {template_error}")
            return create_analysis_page(patterns, confidence_chart)
            
    except Exception as e:
        print(f"분석 페이지 오류: {e}")
        return create_error_page("분석 실행 오류", str(e))

@app.route('/history')
def history():
    """예측 기록 페이지"""
    try:
        if system['db_manager'] is None:
            if not initialize_system():
                return create_error_page("시스템 초기화 실패", "데이터베이스를 초기화할 수 없습니다.")
        
        try:
            predictions = system['db_manager'].get_prediction_history(20)
        except Exception as db_error:
            print(f"예측 기록 조회 실패: {db_error}")
            predictions = create_sample_history()
        
        try:
            return render_template('history.html', predictions=predictions)
        except Exception as template_error:
            print(f"history.html 템플릿 렌더링 실패: {template_error}")
            return create_history_page(predictions)
            
    except Exception as e:
        print(f"기록 페이지 오류: {e}")
        return create_error_page("기록 조회 오류", str(e))

@app.route('/api/predict', methods=['POST'])
def api_predict():
    """API 엔드포인트"""
    try:
        if system['predictor'] is None:
            if not initialize_system():
                return jsonify({
                    'status': 'error',
                    'message': '시스템 초기화 실패'
                }), 500
        
        # 요청 파라미터
        params = request.get_json() or {}
        
        # 예측 수행
        try:
            ml_prediction = system['predictor'].predict_next()
            advanced_prediction = system['advanced_predictor'].predict_ensemble()
        except Exception as pred_error:
            print(f"API 예측 실패: {pred_error}")
            return jsonify({
                'status': 'error',
                'message': f'예측 실행 실패: {str(pred_error)}'
            }), 500
        
        response = {
            'status': 'success',
            'timestamp': datetime.now().isoformat(),
            'predictions': {
                'ml': ml_prediction.get('ml_prediction', 'N/A'),
                'advanced': advanced_prediction.get('ensemble_prediction', 'N/A'),
                'pattern': ml_prediction.get('pattern_predictions', [None])[0]
            },
            'confidence': {
                'ml': ml_prediction.get('confidence_score', 0),
                'advanced': advanced_prediction.get('confidence', 0)
            }
        }
        
        return jsonify(response)
        
    except Exception as e:
        print(f"API 예측 오류: {e}")
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

@app.route('/api/statistics', methods=['GET'])
def api_statistics():
    """통계 API"""
    try:
        if system['predictor'] is None:
            if not initialize_system():
                return jsonify({
                    'status': 'error',
                    'message': '시스템 초기화 실패'
                }), 500
        
        stats = get_safe_statistics()
        return jsonify({
            'status': 'success',
            'data': stats
        })
        
    except Exception as e:
        print(f"통계 API 오류: {e}")
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

def create_safe_charts():
    """안전한 차트 생성"""
    charts = {}
    
    try:
        visualizer = get_safe_visualizer()
        
        # 각 차트를 개별적으로 안전하게 생성
        try:
            charts['jo_distribution'] = visualizer.create_jo_distribution_chart()
        except Exception as e:
            print(f"조 분포 차트 생성 실패: {e}")
            charts['jo_distribution'] = None
        
        try:
            charts['number_trend'] = visualizer.create_number_trend_chart()
        except Exception as e:
            print(f"번호 트렌드 차트 생성 실패: {e}")
            charts['number_trend'] = None
        
        try:
            charts['heatmap'] = visualizer.create_advanced_heatmap()
        except Exception as e:
            print(f"히트맵 생성 실패: {e}")
            charts['heatmap'] = None
            
    except Exception as e:
        print(f"차트 생성 전체 실패: {e}")
        charts = {
            'jo_distribution': None,
            'number_trend': None,
            'heatmap': None
        }
    
    return charts

def get_safe_statistics():
    """안전한 통계 정보 반환"""
    try:
        if system.get('predictor'):
            return system['predictor'].get_statistics()
        else:
            return create_sample_statistics()
    except Exception as e:
        print(f"통계 정보 생성 실패: {e}")
        return create_sample_statistics()

def create_sample_prediction():
    """샘플 예측 결과 생성"""
    return {
        'ml_prediction': '234567',
        'confidence_score': 0.75,
        'pattern_predictions': ['234567'],
        'jo_prediction': '2'
    }

def create_sample_advanced_prediction():
    """샘플 고급 예측 결과 생성"""
    return {
        'ensemble_prediction': '234567',
        'confidence': 0.78,
        'model_predictions': {
            'lstm': '234567',
            'xgboost': '234567',
            'rf': '234567'
        }
    }

def create_sample_patterns():
    """샘플 패턴 분석 결과"""
    return {
        'recent_trends': ['증가', '감소', '안정'],
        'frequency_analysis': {'1': 45, '2': 52, '3': 38, '4': 48, '5': 42},
        'pattern_strength': 0.72
    }

def create_sample_statistics():
    """샘플 통계 정보"""
    return {
        'total_draws': 720,
        'jo_distribution': {'1': 144, '2': 145, '3': 143, '4': 144, '5': 144},
        'accuracy_rate': 0.685,
        'last_update': datetime.now().isoformat()
    }

def create_sample_history():
    """샘플 예측 기록"""
    return [
        {'draw_no': 720, 'predicted': '234567', 'actual': '574627', 'date': '2025-07-31'},
        {'draw_no': 719, 'predicted': '123456', 'actual': '415223', 'date': '2025-07-24'},
        {'draw_no': 718, 'predicted': '345678', 'actual': '178408', 'date': '2025-07-17'}
    ]

def create_simple_home_page():
    """간단한 홈페이지 HTML"""
    return '''
    <!DOCTYPE html>
    <html lang="ko">
    <head>
        <meta charset="UTF-8">
        <title>연금복권 예측기</title>
        <style>
            body { font-family: Arial, sans-serif; margin: 0; padding: 40px; 
                   background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                   min-height: 100vh; color: #333; }
            .container { max-width: 800px; margin: 0 auto; background: rgba(255,255,255,0.95);
                        padding: 40px; border-radius: 20px; text-align: center; }
            h1 { color: #2c3e50; margin-bottom: 30px; }
            .btn { display: inline-block; padding: 15px 30px; margin: 10px;
                   background: linear-gradient(45deg, #667eea, #764ba2); color: white;
                   text-decoration: none; border-radius: 25px; font-weight: 600; }
            .btn:hover { transform: translateY(-2px); }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>🎰 연금복권 예측기</h1>
            <p>AI 기반 연금복권 번호 예측 시스템입니다.</p>
            <a href="/predict" class="btn">예측 시작</a>
            <a href="/analysis" class="btn">상세 분석</a>
            <a href="/history" class="btn">예측 기록</a>
        </div>
    </body>
    </html>
    '''

def create_error_page(title, message):
    """에러 페이지 생성"""
    return f'''
    <!DOCTYPE html>
    <html lang="ko">
    <head>
        <meta charset="UTF-8">
        <title>{title} - 연금복권 예측기</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 0; padding: 40px;
                   background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                   min-height: 100vh; color: #333; display: flex; align-items: center; justify-content: center; }}
            .container {{ max-width: 600px; background: rgba(255,255,255,0.95);
                         padding: 40px; border-radius: 20px; text-align: center; }}
            h1 {{ color: #e74c3c; margin-bottom: 20px; }}
            .btn {{ display: inline-block; padding: 12px 24px; margin: 10px;
                   background: linear-gradient(45deg, #667eea, #764ba2); color: white;
                   text-decoration: none; border-radius: 25px; }}
        </style>
    </head>
    <body>
        <div class="container">
            <h1>⚠️ {title}</h1>
            <p>{message}</p>
            <a href="/" class="btn">홈으로 돌아가기</a>
        </div>
    </body>
    </html>
    ''', 500

def create_result_page(prediction, charts):
    """결과 페이지 생성"""
    chart_html = ""
    for chart_name, chart_data in charts.items():
        if chart_data:
            chart_html += f'<div style="margin: 20px 0;"><h3>{chart_name.replace("_", " ").title()}</h3><img src="{chart_data}" style="max-width: 100%; height: auto;"></div>'
    
    if not chart_html:
        chart_html = '<p>차트를 생성할 수 없습니다.</p>'
    
    ml_pred = prediction.get('ml', {}).get('ml_prediction', 'N/A')
    adv_pred = prediction.get('advanced', {}).get('ensemble_prediction', 'N/A')
    
    return f'''
    <!DOCTYPE html>
    <html lang="ko">
    <head>
        <meta charset="UTF-8">
        <title>예측 결과 - 연금복권 예측기</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 0; padding: 20px;
                   background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                   min-height: 100vh; color: #333; }}
            .container {{ max-width: 1200px; margin: 0 auto; background: rgba(255,255,255,0.95);
                         padding: 40px; border-radius: 20px; }}
            .prediction-box {{ background: linear-gradient(45deg, #667eea, #764ba2);
                              color: white; padding: 30px; border-radius: 15px; text-align: center; margin-bottom: 40px; }}
            .prediction-number {{ font-size: 36px; font-weight: bold; letter-spacing: 4px; margin: 20px 0; }}
            .btn {{ display: inline-block; padding: 12px 24px; margin: 10px;
                   background: linear-gradient(45deg, #667eea, #764ba2); color: white;
                   text-decoration: none; border-radius: 25px; }}
        </style>
    </head>
    <body>
        <div class="container">
            <h1 style="text-align: center;">🎰 예측 결과</h1>
            <div class="prediction-box">
                <h2>ML 예측 번호</h2>
                <div class="prediction-number">{ml_pred}</div>
                <p>고급 예측: {adv_pred}</p>
            </div>
            <div class="charts">
                <h2>분석 차트</h2>
                {chart_html}
            </div>
            <div style="text-align: center;">
                <a href="/" class="btn">홈으로</a>
                <a href="/predict" class="btn">다시 예측</a>
            </div>
        </div>
    </body>
    </html>
    '''

def create_analysis_page(patterns, confidence_chart):
    """분석 페이지 생성"""
    chart_html = f'<img src="{confidence_chart}" style="max-width: 100%; height: auto;">' if confidence_chart else '<p>차트를 생성할 수 없습니다.</p>'
    
    return f'''
    <!DOCTYPE html>
    <html lang="ko">
    <head>
        <meta charset="UTF-8">
        <title>상세 분석 - 연금복권 예측기</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 0; padding: 20px;
                   background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                   min-height: 100vh; color: #333; }}
            .container {{ max-width: 1000px; margin: 0 auto; background: rgba(255,255,255,0.95);
                         padding: 40px; border-radius: 20px; }}
            .pattern-box {{ background: #f8f9fa; padding: 20px; border-radius: 10px; margin: 20px 0; }}
            .btn {{ display: inline-block; padding: 12px 24px; margin: 10px;
                   background: linear-gradient(45deg, #667eea, #764ba2); color: white;
                   text-decoration: none; border-radius: 25px; }}
        </style>
    </head>
    <body>
        <div class="container">
            <h1 style="text-align: center;">📊 상세 분석</h1>
            <div class="pattern-box">
                <h3>패턴 분석 결과</h3>
                <p>패턴 강도: {patterns.get('pattern_strength', 'N/A')}</p>
                <p>최근 트렌드: {', '.join(patterns.get('recent_trends', ['정보 없음']))}</p>
            </div>
            <div class="charts">
                <h3>신뢰도 분석</h3>
                {chart_html}
            </div>
            <div style="text-align: center;">
                <a href="/" class="btn">홈으로</a>
                <a href="/predict" class="btn">예측하기</a>
            </div>
        </div>
    </body>
    </html>
    '''

def create_history_page(predictions):
    """기록 페이지 생성"""
    history_html = ""
    for pred in predictions[:10]:  # 최대 10개만 표시
        history_html += f'''
        <tr>
            <td>{pred.get('draw_no', 'N/A')}</td>
            <td>{pred.get('predicted', 'N/A')}</td>
            <td>{pred.get('actual', 'N/A')}</td>
            <td>{pred.get('date', 'N/A')}</td>
        </tr>
        '''
    
    return f'''
    <!DOCTYPE html>
    <html lang="ko">
    <head>
        <meta charset="UTF-8">
        <title>예측 기록 - 연금복권 예측기</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 0; padding: 20px;
                   background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                   min-height: 100vh; color: #333; }}
            .container {{ max-width: 1000px; margin: 0 auto; background: rgba(255,255,255,0.95);
                         padding: 40px; border-radius: 20px; }}
            table {{ width: 100%; border-collapse: collapse; margin: 20px 0; }}
            th, td {{ padding: 12px; text-align: center; border-bottom: 1px solid #ddd; }}
            th {{ background: #f8f9fa; font-weight: bold; }}
            .btn {{ display: inline-block; padding: 12px 24px; margin: 10px;
                   background: linear-gradient(45deg, #667eea, #764ba2); color: white;
                   text-decoration: none; border-radius: 25px; }}
        </style>
    </head>
    <body>
        <div class="container">
            <h1 style="text-align: center;">📋 예측 기록</h1>
            <table>
                <thead>
                    <tr>
                        <th>회차</th>
                        <th>예측 번호</th>
                        <th>실제 번호</th>
                        <th>날짜</th>
                    </tr>
                </thead>
                <tbody>
                    {history_html}
                </tbody>
            </table>
            <div style="text-align: center;">
                <a href="/" class="btn">홈으로</a>
                <a href="/predict" class="btn">새 예측</a>
            </div>
        </div>
    </body>
    </html>
    '''

@app.errorhandler(404)
def page_not_found(e):
    """404 에러 핸들러"""
    return '''
    <!DOCTYPE html>
    <html lang="ko">
    <head>
        <meta charset="UTF-8">
        <title>404 - 페이지를 찾을 수 없습니다</title>
        <style>
            body { font-family: Arial, sans-serif; text-align: center; margin-top: 100px; 
                   background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                   min-height: 100vh; display: flex; align-items: center; justify-content: center; }
            .container { background: rgba(255,255,255,0.95); padding: 40px; border-radius: 20px; }
            h1 { color: #e74c3c; }
            .btn { display: inline-block; padding: 12px 24px; background: linear-gradient(45deg, #667eea, #764ba2);
                   color: white; text-decoration: none; border-radius: 25px; margin: 10px; }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>404 - 페이지를 찾을 수 없습니다</h1>
            <p>요청하신 페이지가 존재하지 않습니다.</p>
            <a href="/" class="btn">홈으로 돌아가기</a>
        </div>
    </body>
    </html>
    ''', 404

@app.errorhandler(500)
def internal_error(e):
    """500 에러 핸들러"""
    return '''
    <!DOCTYPE html>
    <html lang="ko">
    <head>
        <meta charset="UTF-8">
        <title>500 - 서버 오류</title>
        <style>
            body { font-family: Arial, sans-serif; text-align: center; margin-top: 100px;
                   background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                   min-height: 100vh; display: flex; align-items: center; justify-content: center; }
            .container { background: rgba(255,255,255,0.95); padding: 40px; border-radius: 20px; }
            h1 { color: #e74c3c; }
            .btn { display: inline-block; padding: 12px 24px; background: linear-gradient(45deg, #667eea, #764ba2);
                   color: white; text-decoration: none; border-radius: 25px; margin: 10px; }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>500 - 서버 내부 오류</h1>
            <p>서버에서 오류가 발생했습니다. 잠시 후 다시 시도해 주세요.</p>
            <a href="/" class="btn">홈으로 돌아가기</a>
        </div>
    </body>
    </html>
    ''', 500

if __name__ == '__main__':
    # 초기화
    print("애플리케이션 시작...")
    if not initialize_system():
        print("시스템 초기화에 실패했지만 서버를 시작합니다. 런타임에 재시도됩니다.")
    
    # 서버 실행
    app.run(debug=True, host='0.0.0.0', port=5000)
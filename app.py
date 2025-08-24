from flask import Flask, render_template

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html',
                           title='홈페이지',
                           message='Flask로 만든 간단한 웹사이트입니다!')

@app.route('/about')
def about():
    return render_template('about.html',
                           title='소개',
                           content='이 웹사이트는 Flask 프레임워크로 개발되었습니다.')

@app.route('/contact')
def contact():
    contacts = [
        {'name':'이메일', 'value':'example@email.com'},
        {'name':'전화', 'value':'010-1234-5678'},
        {'name':'주소', 'value':'서울특별시 강남구'}
    ]
    return render_template('contact.html',
                           title='연락처',
                           contacts=contacts)

if __name__=='__main__':
    app.run(debug=True)
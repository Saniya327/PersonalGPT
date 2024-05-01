from flask import Flask

from mygpt import get_query_answer
from mygpt import simple_response
from mygpt import initialize_db

app = Flask(__name__)


app = Flask(__name__)
@app.before_request
def init_db_connection():
   db = initialize_db()



@app.route('/answer')
def index():
    return get_query_answer(db)


if __name__ == '__main__':
    app.run(debug=True)
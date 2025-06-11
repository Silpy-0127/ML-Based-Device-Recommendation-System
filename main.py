from flask import Flask, render_template, request, redirect, flash, session, url_for
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import sqlite3
import os
import ast
import re
import json
from flask import send_file
import io
from xhtml2pdf import pisa
from num2words import num2words

# Load pre-trained Sentence-BERT model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Load data
df = pd.read_pickle('product_embeddings.pkl')

def safe_parse_image(x):
    if isinstance(x, str):
        try:
            return ast.literal_eval(x)
        except Exception:
            return [x]
    return x

df['image'] = df['image'].apply(safe_parse_image)

app = Flask(__name__)
app.secret_key = "4f8c75b73cdd9aab7ff267d948f1e25f"

# Create users table if it doesn't exist
def init_db():
    conn = sqlite3.connect('users.db')
    cur = conn.cursor()
    cur.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            password TEXT NOT NULL
        )
    ''')
    conn.commit()
    conn.close()
init_db()

# Create cart table if it doesn't exist
def init_cart_table():
    conn = sqlite3.connect('users.db')
    cur = conn.cursor()
    cur.execute('''
        CREATE TABLE IF NOT EXISTS user_cart (
            username TEXT,
            cart TEXT
        )
    ''')
    conn.commit()
    conn.close()
init_cart_table()


@app.route('/')
def home():
    return render_template('home.html')


@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        uname = request.form['username']
        pwd = request.form['password']
        conn = sqlite3.connect('users.db')
        cur = conn.cursor()
        cur.execute("SELECT * FROM users WHERE username=? AND password=?", (uname, pwd))
        user = cur.fetchone()

        if user:
            session['username'] = uname
            session['cart'] = "[]"  # This line forces a fresh cart every login

            cur.execute("SELECT cart FROM user_cart WHERE username=?", (uname,))
            cart_row = cur.fetchone()
            if cart_row and cart_row[0]:
                session['cart'] = cart_row[0]  # Store JSON string
            else:
                session['cart'] = "[]"

            conn.close()
            return redirect('/recommend')

        conn.close()
        flash("Invalid credentials")
        return redirect('/login')

    return render_template('login.html')


@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        uname = request.form['username']
        pwd = request.form['password']
        conn = sqlite3.connect('users.db')
        cur = conn.cursor()
        try:
            cur.execute("INSERT INTO users (username, password) VALUES (?, ?)", (uname, pwd))
            conn.commit()
            flash("Registration successful. Please login.")
            return redirect('/login')
        except:
            flash("Username already exists.")
            return redirect('/signup')
        finally:
            conn.close()
    return render_template('signup.html')

@app.route('/logout')
def logout():
    session.clear()  # Use this instead of manually popping keys
    return redirect('/login')



@app.route('/save_cart', methods=['POST'])
def save_cart():
    if 'username' not in session:
        return '', 403

    data = request.get_json()
    cart_json = data.get('cart', '[]')  # already a JSON string

    # Save to DB
    with sqlite3.connect('users.db', timeout=10) as conn:
        cur = conn.cursor()
        cur.execute("DELETE FROM user_cart WHERE username=?", (session['username'],))
        cur.execute("INSERT INTO user_cart (username, cart) VALUES (?, ?)", (session['username'], cart_json))
        conn.commit()
    
    # Also update session cart
    session['cart'] = cart_json

    return '', 204


@app.route('/clear_cart', methods=['POST'])
def clear_cart():
    if 'username' in session:
        data = request.get_json()
        cart_json = data.get('cart', '[]')

        conn = sqlite3.connect('users.db')
        cur = conn.cursor()
        cur.execute("DELETE FROM user_cart WHERE username=?", (session['username'],))
        cur.execute("INSERT INTO user_cart (username, cart) VALUES (?, ?)", (session['username'], cart_json))
        conn.commit()
        conn.close()

        session.pop('cart', None)  # Clear session cart
        session.pop('username', None)

    return '', 204


@app.route('/abstract')
def abstract():
    return render_template('abstract.html')

@app.route('/activity')
def activity():
    return render_template('activity.html')    

@app.route('/future')
def future():
    return render_template('future.html')  


@app.route('/bill')
def bill():
    if 'username' not in session:
        return redirect('/login')

    # Fetch latest cart from DB
    username = session['username']
    with sqlite3.connect('users.db') as conn:
        cur = conn.cursor()
        cur.execute("SELECT cart FROM user_cart WHERE username=?", (username,))
        result = cur.fetchone()
        if result:
            cart_json = result[0]
        else:
            cart_json = '[]'

    try:
        cart_data = json.loads(cart_json)
    except json.JSONDecodeError:
        cart_data = []

    total = sum(item.get('total', 0) for item in cart_data)
    amount_words = num2words(total, to='cardinal', lang='en').capitalize()

    return render_template('bill.html', cart=cart_data, total=total, amount_words=amount_words)


@app.route('/download_pdf', methods=['POST'])
def download_pdf():
    cart_json = session.get('cart', '[]')
    try:
        cart_data = json.loads(cart_json)
    except json.JSONDecodeError:
        cart_data = []

    total = sum(item.get('total', 0) for item in cart_data)
    amount_words = num2words(total, to='cardinal', lang='en').capitalize()

    html = render_template('bill.html', cart=cart_data, total=total, amount_words=amount_words)
    pdf = io.BytesIO()
    pisa_status = pisa.CreatePDF(io.StringIO(html), dest=pdf)

    if pisa_status.err:
        return "PDF generation failed", 500

    pdf.seek(0)
    return send_file(pdf, download_name="bill.pdf", as_attachment=True)


def recommend_products(query, top_k=10):
    query = query.lower()
    query_embedding = model.encode(query)
    filtered_df = df.copy()

    filtered_df['name'] = filtered_df['name'].fillna('').astype(str)
    filtered_df['product'] = filtered_df['product'].fillna('').astype(str)
    filtered_df['main_category'] = filtered_df['main_category'].fillna('').astype(str)
    filtered_df['specifications'] = filtered_df['specifications'].fillna('').astype(str)

    # Remove unrelated items like feature phones, landlines, etc.
    excluded_keywords = ['feature phone', 'keypad', 'landline']
    filtered_df = filtered_df[
        ~filtered_df['main_category'].str.lower().str.contains('|'.join(excluded_keywords)) &
        ~filtered_df['name'].str.lower().str.contains('|'.join(excluded_keywords))
    ]

    # Define category keywords
    category_keywords = {
        'mobile': ['smartphone', 'mobile', 'android', 'ios', '5g', 'iphone'],
        'headphones': ['headphone', 'earbud', 'tws', 'wireless', 'wired', 'bluetooth', 'noise cancelling', 'over-ear', 'airpods'],
        'laptop': ['laptop', 'notebook', 'macbook', 'ultrabook', 'gaming laptop', 'office laptop'],
        'powerbank': ['power bank', 'powerbank', 'battery pack', 'portable charger', '10000mah', '20000mah'],
    }

    category = None
    for cat, keywords in category_keywords.items():
        if any(kw in query for kw in keywords):
            category = cat
            break

    # If no known category found, default to 'mobile'
    if category is None:
        category = 'mobile'

    # Apply filtering based on detected category
    category_filter = category_keywords[category]
    filtered_df = filtered_df[
        filtered_df['name'].str.lower().str.contains('|'.join(category_filter)) |
        filtered_df['specifications'].str.lower().str.contains('|'.join(category_filter)) |
        filtered_df['product'].str.lower().str.contains('|'.join(category_filter)) |
        filtered_df['main_category'].str.lower().str.contains('|'.join(category_filter))
    ]

    # Brand filter (optional enhancement)
    brand_keywords = ['samsung', 'redmi', 'realme', 'motorola', 'vivo', 'oppo', 'oneplus', 'infinix', 'xiaomi',
                      'iqoo', 'lava', 'tecno', 'nokia', 'asus', 'apple', 'boat', 'noise', 'sony', 'jbl', 'hp',
                      'dell', 'lenovo', 'acer', 'msi']
    query_brand = next((b for b in brand_keywords if b in query), None)
    if query_brand:
        filtered_df = filtered_df[
            filtered_df['name'].str.lower().str.contains(query_brand) |
            filtered_df['product'].str.lower().str.contains(query_brand)
        ]

    # Handle price range queries
    if 'actual_price' in filtered_df.columns:
        filtered_df['actual_price'] = (
            filtered_df['actual_price'].astype(str)
            .str.replace(',', '', regex=False)
            .str.extract(r'(\d+)')[0]
        )
        filtered_df['actual_price'] = pd.to_numeric(filtered_df['actual_price'], errors='coerce')

        price_match = re.search(r'under\s*(\d+)', query)
        if price_match:
            max_price = int(price_match.group(1))
            filtered_df = filtered_df[filtered_df['actual_price'] <= max_price]
        else:
            if any(kw in query for kw in ['budget', 'affordable', 'cheap', 'value for money']):
                filtered_df = filtered_df[
                    (filtered_df['actual_price'] >= 10000) & (filtered_df['actual_price'] <= 20000)
                ]
            elif any(kw in query for kw in ['costliest', 'expensive', 'premium', 'high-end', 'flagship']):
                filtered_df = filtered_df[filtered_df['actual_price'] >= 50000]

    # Similarity + ratings
    if 'embeddings' in filtered_df.columns:
        filtered_df['similarity'] = filtered_df['embeddings'].apply(
            lambda x: cosine_similarity([query_embedding], [x]).flatten()[0]
        )
    else:
        filtered_df['similarity'] = 0.5

    if 'ratings' in filtered_df.columns:
        filtered_df['ratings'] = pd.to_numeric(filtered_df['ratings'], errors='coerce')
        filtered_df['ratings_normalized'] = filtered_df['ratings'].fillna(3) / 5.0
    else:
        filtered_df['ratings_normalized'] = 0.5

    filtered_df['score'] = 0.7 * filtered_df['similarity'] + 0.3 * filtered_df['ratings_normalized']

    # Boost based on functional intent (applies to all categories)
    if 'gaming' in query:
        boost_keywords = ['gaming', 'snapdragon', 'turbo', 'cooling', 'g95', 'gtx', 'rtx']
    elif 'camera' in query:
        boost_keywords = ['camera', 'mp', 'ois', 'sony', 'lens', 'imx']
    elif 'battery' in query:
        boost_keywords = ['battery', 'mah', '6000', 'power', 'charge']
    else:
        boost_keywords = []

    if boost_keywords:
        filtered_df['boost'] = filtered_df['specifications'].str.lower().apply(
            lambda x: any(k in x for k in boost_keywords)
        ).astype(int) * 0.2
        filtered_df['score'] += filtered_df['boost']

    filtered_df = filtered_df.sort_values(by='score', ascending=False)
    return filtered_df[['name', 'product', 'actual_price', 'ratings', 'image']].head(top_k)


@app.route('/recommend', methods=['GET', 'POST'])
def recommend():
    if 'username' not in session:
        return redirect('/login')

    recommendations = []
    username = session['username']
    first_letter = username[0].upper()

    # Load cart from session and parse JSON string into Python list
    try:
        cart_data = json.loads(session.get('cart', '[]'))
    except Exception:
        cart_data = []

    if request.method == 'POST':
        query = request.form['query']
        recommendations = recommend_products(query).to_dict(orient='records')

    return render_template('recommend.html',
                           recommendations=recommendations,
                           username=username,
                           first_letter=first_letter,
                           cart_data=cart_data)  # Now it's a list


if __name__ == '__main__':
    app.run(debug=True)

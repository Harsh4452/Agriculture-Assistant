from flask_login import LoginManager, UserMixin, login_user, logout_user, login_required, current_user
from werkzeug.security import generate_password_hash, check_password_hash
class User(UserMixin):
    def __init__(self, id, username, email, password, created_at=None, is_admin=False, db_connection=None):
        self.id = id
        self.username = username
        self.email = email
        self.password = password
        self.created_at = created_at
        self.is_admin = is_admin  # Add admin flag
        self.db = db_connection

    @staticmethod
    def get(user_id, db_connection):
        query = "SELECT * FROM users WHERE id = %s"
        result = db_connection.execute_query(query, (user_id,), fetch=True)
        if result:
            user_data = result[0]
            return User(
                id=user_data['id'],
                username=user_data['username'],
                email=user_data['email'],
                password=user_data['password'],
                created_at=user_data['created_at'],
                is_admin=bool(user_data['is_admin']),  # Convert to boolean
                db_connection=db_connection
            )
        return None

    @staticmethod
    def find_by_email(email, db_connection):
        query = "SELECT * FROM users WHERE email = %s"
        result = db_connection.execute_query(query, (email,), fetch=True)
        if result:
            user_data = result[0]
            return User(
                id=user_data['id'],
                username=user_data['username'],
                email=user_data['email'],
                password=user_data['password'],
                created_at=user_data['created_at'],
                is_admin=bool(user_data['is_admin'])  # Add this
            )
        return None

    @staticmethod
    def find_by_username(username, db_connection):
        query = "SELECT * FROM users WHERE username = %s"
        result = db_connection.execute_query(query, (username,), fetch=True)
        if result:
            user_data = result[0]
            return User(
                id=user_data['id'],
                username=user_data['username'],
                email=user_data['email'],
                password=user_data['password'],
                created_at=user_data['created_at'],
                is_admin=bool(user_data['is_admin'])  # Add this
            )
        return None

    @staticmethod
    def create(username, email, password, is_admin=False, db_connection=None):
        hashed_password = generate_password_hash(password)
        query = "INSERT INTO users (username, email, password, is_admin) VALUES (%s, %s, %s, %s)"
        user_id = db_connection.execute_query(
            query, 
            (username, email, hashed_password, is_admin)
        )
        if user_id:
            return User.get(user_id, db_connection)
        return None
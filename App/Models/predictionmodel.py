class CropPrediction:
    @staticmethod
    def create(db,user_id, nitrogen, phosphorus, potassium, temperature, humidity, ph, rainfall, predicted_crop):
        query = """
        INSERT INTO crop_predictions 
        (user_id, nitrogen, phosphorus, potassium, temperature, humidity, ph, rainfall, predicted_crop)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
        """
        prediction_id = db.execute_query(query, (
            user_id, nitrogen, phosphorus, potassium, 
            temperature, humidity, ph, rainfall, predicted_crop
        ))
        return prediction_id

    @staticmethod
    def get_by_user(db,user_id, limit=None):
        query = "SELECT * FROM crop_predictions WHERE user_id = %s ORDER BY created_at DESC"
        if limit:
            query += f" LIMIT {limit}"
        return db.execute_query(query, (user_id,), fetch=True)

class PricePrediction:
    @staticmethod
    def create(db,user_id, commodity, state, district, market, predicted_price):
        query = """
        INSERT INTO price_predictions 
        (user_id, commodity, state, district, market, predicted_price)
        VALUES (%s, %s, %s, %s, %s, %s)
        """
        prediction_id = db.execute_query(query, (
            user_id, commodity, state, district, market, predicted_price
        ))
        return prediction_id

    @staticmethod
    def get_by_user(db, user_id, limit=None):
        query = "SELECT * FROM price_predictions WHERE user_id = %s ORDER BY created_at DESC"
        params = [user_id]
        if limit:
            query += " LIMIT %s"
            params.append(limit)
        return db.execute_query(query, params, fetch=True)



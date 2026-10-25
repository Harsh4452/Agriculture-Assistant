class YieldPrediction:
    """
    Model class for crop yield predictions.
    Handles database operations for yield predictions.
    """
    
    def __init__(self, id, user_id, crop, crop_year, season, state, area, 
                 production, annual_rainfall, fertilizer, pesticide, yield_value, 
                 created_at=None, updated_at=None):
        """Initialize a YieldPrediction object with the provided parameters."""
        self.id = id
        self.user_id = user_id
        self.crop = crop
        self.crop_year = crop_year
        self.season = season
        self.state = state
        self.area = area
        self.production = production
        self.annual_rainfall = annual_rainfall
        self.fertilizer = fertilizer
        self.pesticide = pesticide
        self.yield_value = yield_value  # Using yield_value instead of yield (Python keyword)
        self.created_at = created_at
        self.updated_at = updated_at
    
    @classmethod
    def create(cls, db, user_id, crop, crop_year, season, state, area, 
               production, annual_rainfall, fertilizer, pesticide, yield_value):
        """
        Create a new yield prediction record in the database.
        
        Args:
            db: Database connection object
            user_id: ID of the user making the prediction
            crop: Name of the crop
            crop_year: Year of the crop cultivation
            season: Growing season (e.g., Kharif, Rabi)
            state: State where the crop is grown
            area: Area of cultivation in hectares
            production: Predicted total production
            annual_rainfall: Annual rainfall in mm
            fertilizer: Fertilizer usage in kg/hectare
            pesticide: Pesticide usage in kg/hectare
            yield_value: Predicted yield in tons/hectare
            
        Returns:
            int: ID of the newly created record, or None if creation failed
        """
        try:
            query = """
            INSERT INTO yield_predictions (
                user_id, crop, crop_year, season, state, area, production,
                annual_rainfall, fertilizer, pesticide, yield
            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            """
            params = (
                user_id, crop, crop_year, season, state, area, production,
                annual_rainfall, fertilizer, pesticide, yield_value
            )
            
            prediction_id = db.execute_query(query, params)
            return prediction_id
            
        except Exception as e:
            print(f"Error creating yield prediction: {e}")
            return None
    
    @classmethod
    def get_by_id(cls, db, prediction_id):
        """
        Get a yield prediction by its ID.
        
        Args:
            db: Database connection object
            prediction_id: ID of the prediction to retrieve
            
        Returns:
            YieldPrediction: Object if found, None otherwise
        """
        try:
            query = "SELECT * FROM yield_predictions WHERE id = %s"
            result = db.execute_query(query, (prediction_id,), fetch=True)
            
            if result and len(result) > 0:
                prediction = result[0]
                return cls(
                    id=prediction['id'],
                    user_id=prediction['user_id'],
                    crop=prediction['crop'],
                    crop_year=prediction['crop_year'],
                    season=prediction['season'],
                    state=prediction['state'],
                    area=prediction['area'],
                    production=prediction['production'],
                    annual_rainfall=prediction['annual_rainfall'],
                    fertilizer=prediction['fertilizer'],
                    pesticide=prediction['pesticide'],
                    yield_value=prediction['yield'],
                    created_at=prediction['created_at'],
                    updated_at=prediction['updated_at']
                )
            return None
            
        except Exception as e:
            print(f"Error retrieving yield prediction: {e}")
            return None
    
    @classmethod
    def get_by_user(cls, db, user_id, limit=None):
        """
        Get yield predictions for a specific user.
        
        Args:
            db: Database connection object
            user_id: ID of the user whose predictions to retrieve
            limit: Maximum number of predictions to retrieve (optional)
            
        Returns:
            list: List of YieldPrediction objects
        """
        try:
            query = "SELECT * FROM yield_predictions WHERE user_id = %s ORDER BY created_at DESC"
            params = (user_id,)
            
            if limit:
                query += " LIMIT %s"
                params = (user_id, limit)
                
            results = db.execute_query(query, params, fetch=True)
            
            predictions = []
            if results:
                for result in results:
                    predictions.append(
                        cls(
                            id=result['id'],
                            user_id=result['user_id'],
                            crop=result['crop'],
                            crop_year=result['crop_year'],
                            season=result['season'],
                            state=result['state'],
                            area=result['area'],
                            production=result['production'],
                            annual_rainfall=result['annual_rainfall'],
                            fertilizer=result['fertilizer'],
                            pesticide=result['pesticide'],
                            yield_value=result['yield'],
                            created_at=result['created_at'],
                            updated_at=result['updated_at']
                        )
                    )
            return predictions
            
        except Exception as e:
            print(f"Error retrieving yield predictions: {e}")
            return []
    
    def to_dict(self):
        """
        Convert the YieldPrediction object to a dictionary.
        
        Returns:
            dict: Dictionary representation of the YieldPrediction
        """
        return {
            'id': self.id,
            'user_id': self.user_id,
            'crop': self.crop,
            'crop_year': self.crop_year,
            'season': self.season,
            'state': self.state,
            'area': self.area,
            'production': self.production,
            'annual_rainfall': self.annual_rainfall,
            'fertilizer': self.fertilizer,
            'pesticide': self.pesticide,
            'yield': self.yield_value,
            'created_at': self.created_at.strftime('%Y-%m-%d %H:%M:%S') if self.created_at else None,
            'updated_at': self.updated_at.strftime('%Y-%m-%d %H:%M:%S') if self.updated_at else None
        }
-----Pipelines

numerical_transformer = SimpleImputer(strategy='median')  # Your code here

# Preprocessing for categorical data
categorical_transformer = Pipeline(steps = [('imputer', SimpleImputer(strategy='most_frequent')),
                                             ('encode', OneHotEncoder(handle_unknown='ignore'))])

# Bundle preprocessing for numerical and categorical data
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_cols),
        ('cat', categorical_transformer, categorical_cols)
    ])

# Define model
model = RandomForestRegressor(n_estimators=100, random_state=0)

from sklearn.model_selection import GridSearchCV
parameters = [{'criterion' : ['mse', 'mae'], 'min_samples_split'
               : [2, 4, 10]} ]

gridsearch = GridSearchCV(estimator = model,
                          param_grid = parameters,
                          scoring = 'neg_mean_squared_error',
                          cv = 10)
grd = Pipeline(steps=[('preprocessor', preprocessor),
                            ('gridsearch', gridsearch)])
grd.fit(X_train, y_train)
print(gridsearch.best_score_)
print(gridsearch.best_params_)

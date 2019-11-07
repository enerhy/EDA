----Opearations with Arrays----

#DataFrame to Array
train = df.iloc[:].values

#Getting Info
np.info(train)
np.mean(X_train, axis = 0).astype(int)
X_train.shape


#print the head
print(train[:5,:])

------Replacing NaN with 

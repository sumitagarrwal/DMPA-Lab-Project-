import pandas as pd
from sklearn.tree import DecisionTreeClassifier # Import Decision Tree Classifier
from sklearn.model_selection import train_test_split # Import train_test_split function
from sklearn import metrics
from sklearn.model_selection import KFold, cross_val_score
import matplotlib.pyplot as plt
from sklearn.tree import export_graphviz
from sklearn.externals.six import StringIO
from IPython.display import Image
import pydotplus

df = pd.read_csv('combined.csv')
df = df[['temperatureHigh', 'temperatureLow', 'humidity', 'precipIntensityMax', 'precipProbability', 'windSpeed', 'cloudCover']]
df.head()
df.shape
X=df.values[:,0:9]
y=df.values[:,6]
y = y/10



X_train, X_test, y_train, y_test = train_test_split( X, y, test_size = 0.2, random_state = 12)
from sklearn.tree import DecisionTreeRegressor
reg_lessF = DecisionTreeRegressor(max_depth=8)
reg_lessF.fit(X_train, y_train)
y_pred_lessF = reg_lessF.predict(X_test)
X_today = [[36,27,0.57,0,0,0.14,0.37]]
plt.plot(y_pred_lessF[:75], label = 'prediction')
plt.plot(y_test[:75], label = 'data')
plt.legend()
plt.show()

scores = -cross_val_score(reg_lessF, X_test, y_test, scoring='neg_mean_absolute_error', cv=60)
print(scores.mean())
from sklearn.metrics import mean_absolute_error
print("The Mean Absolute Error: %.2f " % mean_absolute_error(y_test, reg_lessF.predict(X_test)))
# print(y_pred_lessF)



y_today = reg_lessF.predict(X_today)
print(y_today)
# dot_data = StringIO()
# export_graphviz(reg_lessF, out_file=dot_data,
#                 filled=True, rounded=True,
#                 special_characters=True,feature_names = ['temperatureHigh', 'temperatureLow', 'humidity', 'precipIntensityMax', 'precipProbability', 'windSpeed', 'cloudCover'],class_names=['0','1'])
# graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
# graph.write_png('tree.png')
# Image(graph.create_png())

# import export_graphviz
from sklearn.tree import export_graphviz

# export the decision tree to a tree.dot file
# for visualizing the plot easily anywhere
export_graphviz(reg_lessF, out_file='tree.dot',
                feature_names=['temperatureHigh', 'temperatureLow', 'humidity', 'precipIntensityMax', 'precipProbability', 'windSpeed', 'cloudCover'])
# graph = pydotplus.graph_from_dot_data('tree.dot.getvalue())
# graph.write_png('tree.png')
# Image(graph.create_png())
# dot_data = StringIO()
# export_graphviz(reg_lessF, out_file='dot_data',feature_names = ['temperatureHigh', 'temperatureLow', 'humidity', 'precipIntensityMax', 'precipProbability', 'windSpeed', 'cloudCover'])
# graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
# graph.wr
# Image(graph.create_png())
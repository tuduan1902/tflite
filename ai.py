import numpy as np
import tensorflow as tf
import random

#Assuming you have loaded your new data as new_data_x
new_data_x_R=[]
new_data_x=[]
#manual:
for i in range(85):
   random_number = random.randint(0, 1)
   new_data_x_R.append(random_number)
print(new_data_x_R)
#auto:
new_data_x.append(new_data_x_R)
new_data_x = np.array(new_data_x)
#Load the saved model
loaded_model = tf.keras.models.load_model(r"/content/drive/MyDrive/Đồ Án ra trường Tuong-Du/trained_model3.h5")

#Make predictions on new data
predictions = loaded_model.predict(new_data_x)

#If you want to get the predicted class labels
predicted_labels = np.argmax(predictions, axis=1)

#The variable 'predicted_labels' now contains the predicted class labels for the new data
list_ten_benh = df["Căn bệnh"]
print("predictions  ", predictions)
print(list_ten_benh[predicted_labels])


#Flatten the array
predictions_flat = predictions.flatten()

#Filter values greater than 0.8 and sort them
filtered_predictions = predictions_flat[predictions_flat > 0.8]
sorted_predictions = np.sort(filtered_predictions)[::-1]

#Keep the top 3 values
top_3_predictions = sorted_predictions[:3]

#Create a mask for the top 3 values
mask = np.zeros_like(predictions_flat, dtype=bool)
for value in top_3_predictions:
   mask |= (predictions_flat == value)

#Set values not in the top 3 to 0
predictions_flat[~mask] = 0

#Reshape the array back to its original shape
filtered_predictions = predictions_flat.reshape(predictions.shape)

print("Filtered predictions with only the top 3 values > 0.8 retained:")
print(filtered_predictions)

#Filter out zero values and get their indices
non_zero_indices = np.nonzero(predictions_flat)[0]
non_zero_values = predictions_flat[non_zero_indices]

#Sort the non-zero values in descending order and get the top values' indices
sorted_non_zero_indices = non_zero_indices[np.argsort(non_zero_values)[::-1]]

#Get the top 3 positions or less if fewer than 3 non-zero values exist
top_positions = sorted_non_zero_indices[:3]

print(f"Top positions in the array: {top_positions}")

answer = "qua dữ liệu trên, tôi chuẩn đoán sơ bộ được rằng bạn có thể mắc "
for i in range(len(top_positions)):
   if i == len(top_positions) -1:
      answer = answer + str(list_ten_benh[top_positions[i]]) + " "
   else:
      answer = answer + str(list_ten_benh[top_positions[i]]) + " hoặc "

print(answer)
answer = "qua dữ liệu trên, tôi chuẩn đoán sơ bộ được rằng bạn có thể mắc "
for i in range(len(top_positions)):
   if i == len(top_positions) -1:
      answer = answer + str(list_ten_benh[top_positions[i]]) + " "
   else:
      answer = answer + str(list_ten_benh[top_positions[i]]) + " hoặc "

print(answer)
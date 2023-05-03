# Flight Predictions
- [Introduction and Objectives](#introduction-and-objectives)
- [Methodology](#methodology)
  - [Data Exploration](#data-exploration)
  - [Evaluation Metrics](#evaluation-metrics)
- [The Model](#the-model)
  - [Training the Model](#testing-the-model)
  - [Evaluating the Model](#evaluating-the-model)
  - [Testing the Model](#testing-the-model)
  - [Model Notebook](#model-notebook)
  - [Sample Video](#sample-video)
- [Results](#results)

# Introduction and Objectives
Airline delays can be a major inconvenience for travelers, causing them to miss connections, arrive late to their destination, and disrupt their plans. These delays can occur due to a variety of factors, including weather, mechanical issues, and air traffic congestion. For frequent travelers, delays can be especially frustrating, as they can lead to interrupted plans and increased stress.

To address this issue, we are creating a machine learning model to predict if a given flight would have a delay and if so, approximately how long that delay would be. By analyzing historical data on flight patterns and delays, the model will be able to identify patterns and make predictions about the likelihood and length of delays for future flights. This information will allow travelers to preemptive measures to their travel and plans. This will also allow airlines to proactively take steps to prevent delays, such as adjusting flight schedules or making maintenance repairs before issues arise. Ultimately, this will help improve the experience for travelers and help airliners better prepare for future delays. The machine learning model of this project aims to accurately answer the question **“Will my flight be delayed?”** and if so, **“How long will the delay be?”**

# Methodology
## Data Exploration
Our flight-related data comes from [Kaggle’s 2015 Flight Delays and Cancellation](https://www.kaggle.com/datasets/usdot/flight-delays?select=airports.csv) dataset. This data includes flight information such as airline, airport origin, airport destination, city, state, country, latitude of origin, longitude of origin, and what day the flight occurred. Using information provided in the Kaggle dataset, we can figure out the weather-related data of the flight by using [Climate.gov’s Past Weather by Zip Code](https://www.climate.gov/maps-data/dataset/past-weather-zip-code-data-table) data table. These two datasets combined allow for our model to factor in previously known information such as flight origin and destination, flight time, airport temperature, precipitation, wind speed, and more for the model to predict the delay time.

From the Kaggle dataset, there are approximately 5 million samples to train, evaluate, and test the model on. To obtain weather information about all the flights, it would take a manual process of entering the zip code and year for every location of the flights in the Kaggle dataset and then downloading a CSV file for each area. This would take an unreasonable amount of time. To overcome this, we decided to focus on flight information only coming from Chicago’s O’Hare International Airport (ORD). This airport is a major international airport in a location that has a large span of weather variations from hot summers to harsh winters with extreme cold and snow. Using only data focused on Chicago, there are approximately 285,000 flight samples to use.

Some shortfalls of this data are that all the flight information comes from one year. Ideally, we would be able to have information about all flights from the O’Hare International Airport from a large span of years, or randomly selected flights. Additionally, all the information is about U.S. domestic flights, and due to a large number of sample locations, we had to focus only on the Chicago airport. The model will be most reliable in predicting delays for flights out of Chicago, but it could potentially be generalized to be used for other airports, with less reliability.

## Evaluation Metrics
# The Model
## Training the Model
## Evaluating the Model
## Testing the Model
## Model Notebook
Click the link below to run our notebook in Google Colab.
<!-- Add Colab link here. Sample Colab thing from old project -->
<table align="left">
  <td>
    <a target="_blank" href="https://colab.research.google.com/github/sebriverso/sebriverso.github.io/blob/master/AirlineDelays.ipynb"><img src="https://www.tensorflow.org/images/colab_logo_32px.png" />Run in Google Colab</a>
  </td>
</table>
<br> <br> <br>
## Sample Video
You can view a video of our model here. <!-- Add in the link for the video.-->
# Results
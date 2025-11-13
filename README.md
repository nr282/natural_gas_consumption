The codebase calculates Natural Gas Consumption at a state level using (1) EIA Data, (2) Weather Data and (3) Census Data. Natural Gas Consumption is the most dynamic, critical driver of the Supply and Demand of Natural Gas in the United States. Indeed, since this is the case, accurate prediction of Natural Gas Consumption should move a Natural Gas Trader close to being able to trade Natural Gas, profitably.  

Data Sources currently used are: 
  1. Virginia EIA Consumption: 
    - Link provided here: https://www.eia.gov/dnav/ng/ng_cons_sum_dcu_SVA_m.htm
    - Pulled in automatically from EIA. 
  3. Weather Data:
    - Link provided here: https://pypi.org/project/python-weather/
    - Automatically updated via pyweather.
  5. Census Data:
    - Link provided here: https://www.census.gov/data/tables/time-series/demo/popest/2020s-counties-total.html
    - Not in programtic fashion. 


Analytical Techniques:
  1. Probablistic Programming: Uses pymc3
     - Allows us to correctly model the data up to figuring out the hyperparameters. These techniques are required to avoid the evil's of weather aggregation. Weather aggregation is deterimental to the investment process. 
  2. Global Optimization: Uses Dual Annealing.
    - Used to find the hyperparameters for the statistical model


Roadmap and Agenda:
Currently, I have implemented Virginia (my current state of residence) as the first step in building out the S&D balance. From the training session, it seems that we can achieve a 2.8% accuracy in sample versus a 22% accuracy of the linear regression and/or historical prediction. This is an order of magnitude better. THIS NEEDS TO BE VERIFIED BEFORE WE EXTEND. Only after this has been verified, Spectral Technologies should look to extend this to the rest of the states in the United States. 

Verification Procedure:
  1. Take the training parameters that were trained in 2022 and optimizied against prediction of 2023 for 2024, by adding weather data to the predictions.
  2. Try to predict on new data, which should come out very soon. This would be considered forward testing and should represent the GOLD STANDARD OF VALIDATION.

After verification, I need to implement: 
  1. The other states that are outside of Viringia for both (1) Residential and (2) Consumption.
  2. This requires implementing the same pathways in the code as we have already for Virginia. This means either downloading population data from Census, and piping it into python in the same method used for Virginia or alternatively, using the Census Population API. This will also require searching the long/latt for each county that is represented in the Census data. 
  3. It should be straightforward development work to extend to other states using the csv file approach.
  4. If using the census api, we will have better results, but it will require piping the data into a PopulationData class object. Many aspects of this has already been abstracted, and these abstracts need and should be used. 
  5. We need to then setup an AWS cluster for each pairing of (state, {residential, commercial}) data points.
  6. The ultimate end goal is to make the code found in main.py for Virginia work for all other states. 


Discussion:

Training the models can take up to a month. This was a key feature, not drawback of the techniques and methods we aimed to employ here. The increased training time allows us to develop a more sophisticated statistical model with unknown hyperparameters and then find the hyperparameters with a global optimization technique. The increased model complexity and appropiate model of the data has the ability to create better predictions than the linear regression techniques employed at Millenium, Brevan Howard, Tudor Investment Corporation etc. Furthermore, the business strategy of using the available training time to compute more sophisticated statistical models than what the current hedge funds currently have is a step in the direction of the future. The improved results found in Natural Gas Trading can be used as an example to investors and other technology companies that Spectral Technologies capabilities are unparalled. 

Task List:
  1. Validate Virginia Residential Consumption
  2. Look to verify against out-of-sample-2024 numbers with parameters trained in the log.
  3. Look to extend the methodology to additional states. 












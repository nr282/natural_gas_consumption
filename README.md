The codebase calculates Natural Gas Consumption at a state level using (1) EIA Data, (2) Weather Data and (3) Census Data. Natural Gas Consumption is the most dynamic, critical driver of the Supply and Demand of Natural Gas in the United States. Indeed, since this is the case, accurate prediction of Natural Gas Consumption should move a Natural Gas Trader close to being able to trade Natural Gas. 

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
     - Allows us to correctly model the data up to figuring out the hyperparameters. These techniques are required to avoid the evil's of weather aggregation. 
  2. Global Optimization: Uses Dual Annealing.
    - Used to find the hyperparameters for the statistical model

Training the models can take up to a month. This was a key feature, not drawback of the techniques and methods we aimed to employ here. The increased training time allows us to develop a more sophisticated statistical model with unknown hyperparameters and then find the hyperparameters with a global optimization technique. The increased model complexity and appropiate model of the data has the ability to create better predictions than the linear regression techniques employed at Millenium, Brevan Howard, Tudor Investment Corporation etc. Furthermore, the business strategy of using the available training time to compute more sophisticated statistical models than what the current hedge funds currently have is a step in the direction of the future. The improved results found in Natural Gas Trading can be used as an example to investors and other technology companies that Spectral Technologies capabilities are unparalled. 







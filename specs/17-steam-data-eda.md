# Exploratory Data Analysis on Steam Interactions Dataset
So far, the features used for the modeling and actual data processing jobs have been more like a demo / hunch style features, 
without no evidence they actually help the model (though some will actually and factually be correct).
The purpose of EDA is to confirm and surface important features that we are probably leaving behind in the two tower (and possibly ranking) 
models.

The deliverable for this spec should include a Jupyter Notebook analyzing interactions data to answer these questions:
- Do most heavy users review games only in a specific cluster of genres / tags?
- Can review reacts (voted useful , voted funny) be used to weight the interactions?
- Does game age influence in a user's review?
- Is the cost of a videogame related to the number of reviews and its popularity (free games)?
- is user recent history and its ordering influencial for next interaction or there is no preferred order? (thinking about using attention)
- Can game text description embedding help in clustering games with similar argument and overall in game features?
- Are negative reviews an important signal for the model?
- Can a sentiment analysis on the top reviews for a game help as another feature for it?

PLease do include other questions you might as well think are important to decide features here.

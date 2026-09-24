## Spec 5: RecSys Serving
Simple Lambda function to serve the recommendations for a user.
Given the recs table already exists from Spec 4, the lambda is just a retriever of the data as an API for the 
requesting user. Auth type of the lambda must be configurable on the infrastructure CD workflow so it can be open for 
everyone on demo or only auth by an appropriate role.

Nice to have: Include the full item features. Currently the games_features table doesn't contain the 
game description, or even any human readable text for returning. So a new table should be useful to store that, along 
with the image url which was also scraped, to be consumed by a to-be-implemented frontend.
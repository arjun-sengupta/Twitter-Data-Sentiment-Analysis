The application leverages a variety of technologies to visualize a real-time stream of twitter data 
using sentiment analysis and word vector space mapping via word2vec.  
It runs on a local Flask instance with a Redis/Celery back-end and uses Socket-IO to push events to the web client.  
Twitter integration is managed by the pattern library.  NVD3 is used on the front-end to create the visualization.

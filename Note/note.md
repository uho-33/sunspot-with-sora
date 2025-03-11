### Token

#### Strategy 1
Consider a timeseries with 8 timestep(4 days) as a token

shortage:
- If in training, we just use one token, label free training would be meaningless.

#### Strategy 2
One data is a token. Using a real or a high dimentonal vector as its embedding.

Adv.:
- In training, label free training can help model find connection among different time.
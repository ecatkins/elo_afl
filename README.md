# ELO AFL Rankings
The [Elo rating system] (https://en.wikipedia.org/wiki/Elo_rating_system) is a method for calculating the relative skill levels of players in competitor-versus-competitor games such as chess.

Inspired by the work of FiveThirtyEight that calculate elo ratings for [NFL] (http://projects.fivethirtyeight.com/complete-history-of-the-nfl/) and NBA, I have used information available [online] (https://doubleclix.wordpress.com/2015/01/20/the-art-of-nfl-ranking-the-elo-algorithm-and-fivethirtyeight/) to derive my own elo system for the Australian Football League (AFL).

### Features
* Tracks ELO ratings for the AFL from 2011-2015
* ELO ratings can be used to estimate the win probabilities for a given game
* I have created a custom hyper-parameter optimisation using cross-log entropy (predicted win probability versus actual result) to adjust the parameters used by FiveThirtyEight for NFL as these are different for every sport
* Charting method to visualise historical elo ratings

### Instructions
```python
x = 'testing'
```

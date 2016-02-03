# ELO AFL Rankings
The [Elo rating system] (https://en.wikipedia.org/wiki/Elo_rating_system) is a method for calculating the relative skill levels of players in competitor-versus-competitor games such as chess.

Inspired by the work of FiveThirtyEight that calculate elo ratings for [NFL] (http://projects.fivethirtyeight.com/complete-history-of-the-nfl/) and NBA, I have used information available [online] (https://doubleclix.wordpress.com/2015/01/20/the-art-of-nfl-ranking-the-elo-algorithm-and-fivethirtyeight/) to derive my own elo system for the Australian Football League (AFL).

Data Source = [Australia Sports Betting] (http://www.aussportsbetting.com/data/historical-afl-results-and-odds-data/)

### Features
* Tracks ELO ratings for the AFL from 2011-2015
* ELO ratings can be used to estimate the win probabilities for a given game
* I have created a custom hyper-parameter optimisation using cross-log entropy (predicted win probability versus actual result) to adjust the parameters used by FiveThirtyEight for NFL as these are different for every sport
* Charting method to visualise historical elo ratings

### Instructions
* Clone directory
* Import module and add data (can replace with latest)
```python
import elo_afl as e
elo = e.Elo()
elo.add_data('afl.xlsx')
```
* Run model and plot historical elo ratings
```python
import elo_afl as e
elo = e.Elo()
elo.run_show(k=16,home_field=220,mean_reversion = 0.6, margin_beta_range = 20)
```
* Test hyper-parameter combinations





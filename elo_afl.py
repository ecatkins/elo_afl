import numpy as np 
import pandas as pd
import seaborn as sns

from sklearn.metrics import log_loss


class Elo:
	
	def __init__(self):
		pass

	def add_data(self, file):
		'''File sourced from 
		http://www.aussportsbetting.com/data/historical-afl-results-and-odds-data/
		'''
		initial_data = pd.read_excel(file)
		self.original_df = self.add_columns(initial_data)
		# print(self.original_df)
		return self.original_df

	def add_columns(self, initial_data):
		initial_data = initial_data.sort(['Date'])
		initial_data['home_odds_estimate'] = 0
		initial_data['away_odds_estimate'] = 0
		initial_data['home_prob'] = 0
		initial_data['away_prob'] = 0
		initial_data['home_result'] = 0
		initial_data['away_result'] = 0 
		initial_data['home_rank'] = 0 
		initial_data['away_rank'] = 0
		return initial_data

	def reset_ratings(self):
		self.current_ratings = {"Adelaide":1500,"Brisbane":1500,"Carlton":1500,"Collingwood":1500,"Essendon":1500,"Fremantle":1500,"Geelong":1500,"Gold Coast":1500,"GWS Giants":1500,"Hawthorn":1500,"Melbourne":1500,"North Melbourne":1500,"Port Adelaide":1500,"Richmond":1500, "St Kilda":1500,"Sydney":1500,"West Coast":1500,"Western Bulldogs":1500}
		self.historical_ratings = {"Adelaide":{"Date":[],"Elo":[]},"Brisbane":{"Date":[],"Elo":[]},"Carlton":{"Date":[],"Elo":[]},"Collingwood":{"Date":[],"Elo":[]},"Essendon":{"Date":[],"Elo":[]},"Fremantle":{"Date":[],"Elo":[]},"Geelong":{"Date":[],"Elo":[]},"Gold Coast":{"Date":[],"Elo":[]},"GWS Giants":{"Date":[],"Elo":[]},"Hawthorn":{"Date":[],"Elo":[]},"Melbourne":{"Date":[],"Elo":[]},"North Melbourne":{"Date":[],"Elo":[]},"Port Adelaide":{"Date":[],"Elo":[]},"Richmond":{"Date":[],"Elo":[]}, "St Kilda":{"Date":[],"Elo":[]},"Sydney":{"Date":[],"Elo":[]},"West Coast":{"Date":[],"Elo":[]},"Western Bulldogs":{"Date":[],"Elo":[]}}

	def initial_rank(self, row, home_team, away_team, mean_reversion):
		''' Calculates the current ranks of the teams including update
			for preseason mean reversion if required
		Args:
			row: current row from match dataframe
			home_team: home team name
			away_team: away team name
			mean_reversion: mean reversion factor for each team after the season ends
		Returns:
			(home_rank, away_rank) -> Current home and away team ranks
		'''

		if row['Round']  == 1:
			home_rank = 1500 + (self.current_ratings[home_team] - 1500) * mean_reversion
			self.current_ratings[home_team] = home_rank
			away_rank = 1500 + (self.current_ratings[away_team] - 1500) * mean_reversion
			self.current_ratings[away_team] = away_rank
		else:
			home_rank = self.current_ratings[home_team]
			away_rank = self.current_ratings[away_team]
		return home_rank, away_rank

	def implied_odds(self,home_rank, away_rank, row_index, df):
		''' Calculates implied probabilites and adds to dataframe
		Args: 
			home_rank: current rank of the home team
			away_rank: current rank of the away team
			row_index: row index in dataframe for current match
			df: current match dataframe
		Returns:
			(prob_home, prob_away, df) -> implied match probabilites for 
			home and away team and updated dataframe
		'''
		# Pr(A) = 1 / (10^(-ELODIFF/400) + 1)
		prob_home = 1 / (10**((away_rank-home_rank)/400)+1)
		prob_away = 1 / (10**((home_rank-away_rank)/400)+1)
		df.ix[row_index,'home_prob'] = prob_home
		df.ix[row_index,'away_prob'] = prob_away
		df.ix[row_index,'home_odds_estimate'] = 1 / prob_home
		df.ix[row_index,'away_odds_estimate'] = 1 / prob_away
		return prob_home, prob_away, df

	def binary_result(self, row_index, df, home_score, away_score):
		''' Calculates binary result for each match and adds to dataframe
		Args:
			row_index: row index in dataframe for current match
			df: current match dataframe
			home_score: match score for home team
			away_score: match score for away team
		Returns: 
			(home_result, away_result, df) -> binary indicating whether a 
			team has won that match, 0.5 for a draw and updated dataframe
		'''

		if home_score > away_score:
			df.ix[row_index,'home_result'] = 1
			df.ix[row_index,'away_result'] = 0
			return 1, 0, df
		elif home_score == away_score:
			df.ix[row_index,'home_result'] = 0.5
			df.ix[row_index,'away_result'] = 0.5
			return 0.5, 0.5, df
		else:
			df.ix[row_index,'home_result'] = 0
			df.ix[row_index,'away_result'] = 1
			return 0, 1, df

	def winners_rank(self, home_result, home_rank, away_rank):
		''' Assigns the winning team as home or away 
		Args:
			home_result: binary indicator of home team result
			home_rank: current rank of the home team
			away_rank: current rank of the away team
		Returns:
			(winners_rank, losers_rank) -> the current rank of 
			the winning and losing team
		'''
		if home_result == 1:
			winners_rank, losers_rank = home_rank, away_rank
		else:
			winners_rank, losers_rank = away_rank, home_rank
		return winners_rank, losers_rank


	def assign_new_rankings(self, row_index, row,  df, home_team, new_home_rank, away_team, new_away_rank):
		''' Assigns calculated new rankings to dictionaries and dataframe
		Args:
			row_index: row index in dataframe for current match
			row: row in dataframe for current match
			home_team: home team name
			new_home_rank: new calculated ranking for home team
			away_team: away team name
			new_away_ rank: new calculated ranking for away team
		Returns:
			Updated dataframe
		'''

		# Assign home team
		self.current_ratings[home_team] = new_home_rank
		self.historical_ratings[home_team]['Date'].append(row['Date'])
		self.historical_ratings[home_team]['Elo'].append(new_home_rank)
		df.ix[row_index,'home_rank'] = new_home_rank
		# Assign away team
		self.current_ratings[away_team] = new_away_rank
		self.historical_ratings[away_team]['Date'].append(row['Date'])
		self.historical_ratings[away_team]['Elo'].append(new_away_rank)
		df.ix[row_index,'away_rank'] = new_away_rank

		return df


	def run_model(self, k, home_field, mean_reversion,margin_smoothing):
		'''  Calculates historical and current elo ratings
		Args: 
			k: the main levearge point to customise the algorithm for different domains
			home_field: the point advantage given to the home team
			mean_reversion: mean reversion factor for each team after the season ends
			margin_smoothing: smoothing parameter for the margin multiplier

		Returns: 
			The updated data frame with elo values and implied odds
		'''
		self.reset_ratings()
		df = self.original_df.copy()
		
		# Loop through each match in sequential order
		for row_index, row in df.iterrows():
			
			# General information
			home_team = row['Home Team']
			away_team = row['Away Team']
			home_score = row['Home Score']
			away_score = row['Away Score']
			point_differential = home_score - away_score
			home_rank, away_rank = self.initial_rank(row, home_team, away_team, mean_reversion)
			
			# Estimate and assign implied odds
			home_prob, away_prob, df = self.implied_odds(home_rank, away_rank, row_index, df)

			# Calculate and assign binary result
			home_result, away_result, df = self.binary_result(row_index, df, home_score, away_score)

			# Assign rank to winners
			winners_rank, losers_rank = self.winners_rank(home_result, home_rank, away_rank)

			# Calculate new rank
			new_home_rank = self.elo_calculation(k,point_differential,home_rank,away_rank, winners_rank,losers_rank,home_result, home_field,margin_smoothing)
			new_away_rank = self.elo_calculation(k,point_differential,away_rank,home_rank, winners_rank,losers_rank,away_result, -home_field,margin_smoothing)

			# Assign new rankings
			df = self.assign_new_rankings(row_index, row, df, home_team, new_home_rank, away_team, new_away_rank)

		self.elo_df = df

		return self.elo_df

	def elo_calculation(self, k, point_differential,old_rank,opponent_rank,winners_rank,losers_rank,actual_result,home_field_advantage,margin_smoothing):
		if winners_rank == old_rank and home_field_advantage > 0:
			winners_advantage = home_field_advantage
		elif winners_rank == opponent_rank and home_field_advantage < 0:
			winners_advantage = -home_field_advantage
		elif winners_rank == old_rank and home_field_advantage < 0:
			winners_advantage = -home_field_advantage
		elif winners_rank == opponent_rank and home_field_advantage > 0:
			winners_advantage = home_field_advantage
		

		margin_multiplier = np.log(abs(point_differential)+1) * margin_smoothing / (0.001*(winners_rank + winners_advantage - losers_rank) + margin_smoothing)
		expected_result = 1 / (1 + 10**((opponent_rank-(old_rank+home_field_advantage))/400))
		new_rank = old_rank + k * margin_multiplier * (actual_result - expected_result)
		return new_rank

	def run_plot(self,k,home_field,mean_reversion,margin_smoothing,teams=False):
		import matplotlib.pyplot as plt

		elo_df = self.run_model(k,home_field,mean_reversion,margin_smoothing)
		elo_ratings = self.current_ratings
		elo_historical = self.historical_ratings

		if teams:
			for team in teams:
				plt.plot(elo_historical[team]['Date'],elo_historical[team]['Elo'],label=team)
				plt.legend(loc=9, bbox_to_anchor=(0.5, -0.1), ncol=4)
		else:
			for team in elo_historical:
				plt.plot(elo_historical[team]['Date'],elo_historical[team]['Elo'],label=team)
				plt.legend(loc=9, bbox_to_anchor=(0.5, -0.1), ncol=4)
		
		count = 0
		for team in elo_historical:
			count += elo_ratings[team]
		print(count/18)

		plt.show()

	def test_parameters(self, k_range, home_field_range, mean_reversion_range, margin_smoothing_range):
		
		best = None
		best_score = 10000000

		for k in k_range:
			for home_field in home_field_range:
				for mean_reversion in mean_reversion_range:
					for margin_smoothing in margin_smoothing_range:
						elo_df = self.run_model(k, home_field, mean_reversion, margin_smoothing)
						not_draw = elo_df['home_result'] != 0.5
						actual = list(elo_df['home_result'][not_draw])
						predicted = list(elo_df['home_prob'][not_draw])
						error = log_loss(actual, predicted)
						if error < best_score:
							best_score = error
							best = k, home_field, mean_reversion, margin_smoothing
						print(k, home_field, mean_reversion, margin_smoothing)
						print(error)
		print("BEST HYPERPARAMETER SET")
		print(best)
		print(best_score)


my_elo = Elo()

my_elo.add_data('afl.xlsx')

my_elo.run_plot(16,110,0.70,12.2,teams=['Fremantle','Melbourne','West Coast'])

# my_elo.run_plot(16,200,0.6,20)

# my_elo.test_parameters(k_range = [14,16,18], home_field_range = [180,200,220],mean_reversion_range=[0,56, 0.58],margin_smoothing_range = [19,20,21,22])









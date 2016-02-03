import numpy as np 
import pandas as pd
import seaborn as sns

from sklearn.metrics import log_loss


class Elo:
	
	def __init__(self):
		self.current_ratings = {"Adelaide":1500,"Brisbane":1500,"Carlton":1500,"Collingwood":1500,"Essendon":1500,"Fremantle":1500,"Geelong":1500,"Gold Coast":1500,"GWS Giants":1500,"Hawthorn":1500,"Melbourne":1500,"North Melbourne":1500,"Port Adelaide":1500,"Richmond":1500, "St Kilda":1500,"Sydney":1500,"West Coast":1500,"Western Bulldogs":1500}
		self.historical_ratings = {"Adelaide":{"Date":[],"Elo":[]},"Brisbane":{"Date":[],"Elo":[]},"Carlton":{"Date":[],"Elo":[]},"Collingwood":{"Date":[],"Elo":[]},"Essendon":{"Date":[],"Elo":[]},"Fremantle":{"Date":[],"Elo":[]},"Geelong":{"Date":[],"Elo":[]},"Gold Coast":{"Date":[],"Elo":[]},"GWS Giants":{"Date":[],"Elo":[]},"Hawthorn":{"Date":[],"Elo":[]},"Melbourne":{"Date":[],"Elo":[]},"North Melbourne":{"Date":[],"Elo":[]},"Port Adelaide":{"Date":[],"Elo":[]},"Richmond":{"Date":[],"Elo":[]}, "St Kilda":{"Date":[],"Elo":[]},"Sydney":{"Date":[],"Elo":[]},"West Coast":{"Date":[],"Elo":[]},"Western Bulldogs":{"Date":[],"Elo":[]}}
		self.original_df = None

	def add_data(self, file):
		initial_data = pd.read_excel(file)
		self.original_df = self.add_columns(initial_data)
		# print(self.original_df)
		return self.original_df

	def add_columns(self, initial_data):
		initial_data = initial_data[0:1300]
		initial_data = initial_data.sort(['Date'])

		initial_data['home_odds_estimate'] = 0
		initial_data['away_odds_estimate'] = 0
		initial_data['home_prob'] = 0
		initial_data['away_prob'] = 0
		initial_data['home_result'] = 0
		initial_data['away_result'] = 0 
		return initial_data

	def reset_ratings(self):
		self.current_ratings = {"Adelaide":1500,"Brisbane":1500,"Carlton":1500,"Collingwood":1500,"Essendon":1500,"Fremantle":1500,"Geelong":1500,"Gold Coast":1500,"GWS Giants":1500,"Hawthorn":1500,"Melbourne":1500,"North Melbourne":1500,"Port Adelaide":1500,"Richmond":1500, "St Kilda":1500,"Sydney":1500,"West Coast":1500,"Western Bulldogs":1500}
		self.historical_ratings = {"Adelaide":{"Date":[],"Elo":[]},"Brisbane":{"Date":[],"Elo":[]},"Carlton":{"Date":[],"Elo":[]},"Collingwood":{"Date":[],"Elo":[]},"Essendon":{"Date":[],"Elo":[]},"Fremantle":{"Date":[],"Elo":[]},"Geelong":{"Date":[],"Elo":[]},"Gold Coast":{"Date":[],"Elo":[]},"GWS Giants":{"Date":[],"Elo":[]},"Hawthorn":{"Date":[],"Elo":[]},"Melbourne":{"Date":[],"Elo":[]},"North Melbourne":{"Date":[],"Elo":[]},"Port Adelaide":{"Date":[],"Elo":[]},"Richmond":{"Date":[],"Elo":[]}, "St Kilda":{"Date":[],"Elo":[]},"Sydney":{"Date":[],"Elo":[]},"West Coast":{"Date":[],"Elo":[]},"Western Bulldogs":{"Date":[],"Elo":[]}}



	def run_model(self, k,home_field,mean_reversion,margin_beta):
		self.reset_ratings()
		count = 0
		df = self.original_df.copy()
		
		for row_index, row in df.iterrows():
			home_team = row['Home Team']
			away_team = row['Away Team']


		
			# Preseason reversion to the mean
			if row['Round'] == 1:
				new_home_ranking = 1500 + (self.current_ratings[home_team] - 1500) * mean_reversion
				self.current_ratings[home_team] = new_home_ranking
				new_away_ranking = 1500 + (self.current_ratings[away_team] - 1500) * mean_reversion
				self.current_ratings[away_team] = new_away_ranking
		
			# General
			home_score = row['Home Score']
			away_score = row['Away Score']
			home_ranking = self.current_ratings[home_team]
			away_ranking = self.current_ratings[away_team]

			# Estimate odds for games
			# Pr(A) = 1 / (10^(-ELODIFF/400) + 1)
			prob_home = 1 / (10**((away_ranking-home_ranking)/400)+1)
			prob_away = 1 / (10**((home_ranking-away_ranking)/400)+1)
			df.ix[row_index,'home_prob'] = prob_home
			df.ix[row_index,'away_prob'] = prob_away
			df.ix[row_index,'home_odds_estimate'] = 1 / prob_home
			df.ix[row_index,'away_odds_estimate'] = 1 / prob_away
			
			# Assign actual result
			if home_score > away_score:
				df.ix[row_index,'home_result'] = 1
				df.ix[row_index,'away_result'] = 0
			elif home_score == away_score:
				df.ix[row_index,'home_result'] = 0.5
				df.ix[row_index,'away_result'] = 0.5
			else:
				df.ix[row_index,'home_result'] = 0
				df.ix[row_index,'away_result'] = 1

			point_differential = home_score - away_score
			if home_score > away_score:
				winners_rank = home_ranking
				losers_rank = away_ranking
			else:
				winners_rank = away_ranking
				losers_rank = home_ranking

			# Home team calculation
			old_rank = home_ranking
			opponent_rank = away_ranking
			home_field_advantage = home_field
			if home_score > away_score:
				actual_result = 1
			elif home_score == away_score:
				actual_result = 0.5
			else:
				actual_result = 0
			new_rank = self.elo_calculation(k,point_differential,old_rank,opponent_rank, winners_rank,losers_rank,actual_result, home_field_advantage,margin_beta)
			self.current_ratings[home_team] = new_rank
			self.historical_ratings[home_team]['Date'].append(row['Date'])
			self.historical_ratings[home_team]['Elo'].append(new_rank)

			# Away team calculation
			old_rank = away_ranking
			opponent_rank = home_ranking
			home_field_advantage = - home_field
			if away_score > home_score:
				actual_result = 1
			elif away_score == home_score:
				actual_result = 0.5
			else:
				actual_result = 0
			new_rank = self.elo_calculation(k,point_differential,old_rank,opponent_rank, winners_rank,losers_rank,actual_result, home_field_advantage,margin_beta)
			self.current_ratings[away_team] = new_rank
			self.historical_ratings[away_team]['Date'].append(row['Date'])
			self.historical_ratings[away_team]['Elo'].append(new_rank)

		self.elo_df = df

		return self.elo_df

	def elo_calculation(self, k, point_differential,old_rank,opponent_rank,winners_rank,losers_rank,actual_result,home_field_advantage,margin_beta):

	
		if winners_rank == old_rank and home_field_advantage > 0:
			winners_advantage = home_field_advantage
		elif winners_rank == opponent_rank and home_field_advantage < 0:
			winners_advantage = -home_field_advantage
		elif winners_rank == old_rank and home_field_advantage < 0:
			winners_advantage = -home_field_advantage
		elif winners_rank == opponent_rank and home_field_advantage > 0:
			winners_advantage = home_field_advantage
		

		margin_multiplier = np.log(abs(point_differential)+1) * margin_beta / (0.001*(winners_rank + winners_advantage - losers_rank) + margin_beta)
		expected_result = 1 / (1 + 10**((opponent_rank-(old_rank+home_field_advantage))/400))
		new_rank = old_rank + k * margin_multiplier * (actual_result - expected_result)
		return new_rank

	def run_show(self,k,home_field,mean_reversion,margin_beta,teams=False):
		import matplotlib.pyplot as plt

		elo_df = self.run_model(k,home_field,mean_reversion,margin_beta)
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

	def test_parameters(self, k_range, home_field_range, mean_reversion_range, margin_beta_range):
		
		best = None
		best_score = 10000000

		for k in k_range:
			for home_field in home_field_range:
				for mean_reversion in mean_reversion_range:
					for margin_beta in margin_beta_range:
						elo_df = self.run_model(k, home_field, mean_reversion, margin_beta)
						not_draw = elo_df['home_result'] != 0.5
						actual = list(elo_df['home_result'][not_draw])
						predicted = list(elo_df['home_prob'][not_draw])
						error = log_loss(actual, predicted)
						if error < best_score:
							best_score = error
							best = k, home_field, mean_reversion, margin_beta
						print(k, home_field, mean_reversion, margin_beta)
						print(error)
		print("BEST HYPERPARAMETER SET")
		print(best)
		print(best_score)


my_elo = Elo()

my_elo.add_data('afl.xlsx')

#  my_elo.run_show(16,110,0.70,12.2,teams=['Fremantle','Hawthorn','West Coast'])

my_elo.run_show(16,200,0.6,20)

# my_elo.test_parameters(k_range = [14,16,18], home_field_range = [180,200,220],mean_reversion_range=[0,56, 0.58],margin_beta_range = [19,20,21,22])









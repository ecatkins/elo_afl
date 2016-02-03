import numpy as np 
import pandas as pd
import seaborn as sns

afl_results = pd.read_excel('afl.xlsx')
afl_results = afl_results[0:1300]
afl_results = afl_results.sort(['Date'])


afl_results['home_odds_estimate'] = 0
afl_results['away_odds_estimate'] = 0
afl_results['home_prob'] = 0
afl_results['away_prob'] = 0
afl_results['home_result'] = 0
afl_results['away_result'] = 0
 # Home field advantage = 65 Elo points in NFL (2.6 points)

 # Pr(A) = 1 / (10^(ELODIFF/400) + 1)

 #Margin of victory
 # Add one point to the team's margin of vicotry and take the natural logarithm and times by the k value

 #Margin of victory multiplier (to discount for favourites wins)
 # = LN(ABS(PD)+1) * (2.2/((ELOW-ELOL)*0.001+2.2))
# PD = Points differential
# ELOW is the winning team's ELO
# ELOL is the losing teams's ELO

# Reversion to the mean of 1/3 post season

# https://doubleclix.wordpress.com/2015/01/20/the-art-of-nfl-ranking-the-elo-algorithm-and-fivethirtyeight/



def elo_calculation(k,point_differential,old_rank,opponent_rank,winners_rank,losers_rank,actual_result,home_field_advantage,margin_beta):
	#Maybe add in home field advantage to both below
	if winners_rank == old_rank and home_field_advantage > 0:
		winners_advantage = home_field_advantage
	elif winners_rank == opponent_rank and home_field_advantage < 0:
		winners_advantage = -home_field_advantage
	elif winners_rank == old_rank and home_field_advantage < 0:
		winners_advantage = -home_field_advantage
	elif winners_rank == opponent_rank and home_field_advantage > 0:
		winners_advantage = home_field_advantage
	

	margin_multiplier = np.log(abs(point_differential)+1) * margin_beta / (0.001*(winners_rank + home_field_advantage - losers_rank) + margin_beta)
	expected_result = 1 / (1 + 10**((opponent_rank-(old_rank+home_field_advantage))/400))
	new_rank = old_rank + k * margin_multiplier * (actual_result - expected_result)
	return new_rank



def afl_elo_timeseries(data_frame,k,home_field,mean_reversion,margin_beta):


	elo_ratings = {"Adelaide":1500,"Brisbane":1500,"Carlton":1500,"Collingwood":1500,"Essendon":1500,"Fremantle":1500,"Geelong":1500,"Gold Coast":1500,"GWS Giants":1500,"Hawthorn":1500,"Melbourne":1500,"North Melbourne":1500,"Port Adelaide":1500,"Richmond":1500, "St Kilda":1500,"Sydney":1500,"West Coast":1500,"Western Bulldogs":1500}

	elo_historical = {"Adelaide":{"Date":[],"Elo":[]},"Brisbane":{"Date":[],"Elo":[]},"Carlton":{"Date":[],"Elo":[]},"Collingwood":{"Date":[],"Elo":[]},"Essendon":{"Date":[],"Elo":[]},"Fremantle":{"Date":[],"Elo":[]},"Geelong":{"Date":[],"Elo":[]},"Gold Coast":{"Date":[],"Elo":[]},"GWS Giants":{"Date":[],"Elo":[]},"Hawthorn":{"Date":[],"Elo":[]},"Melbourne":{"Date":[],"Elo":[]},"North Melbourne":{"Date":[],"Elo":[]},"Port Adelaide":{"Date":[],"Elo":[]},"Richmond":{"Date":[],"Elo":[]}, "St Kilda":{"Date":[],"Elo":[]},"Sydney":{"Date":[],"Elo":[]},"West Coast":{"Date":[],"Elo":[]},"Western Bulldogs":{"Date":[],"Elo":[]}}

	


	


	for row_index, row in data_frame.iterrows():
		home_team = row['Home Team']
		away_team = row['Away Team']

		#Preseason reversion to the mean
		if row['Round'] == 1:
			new_home_ranking = 1500 + (elo_ratings[home_team] - 1500) * mean_reversion
			elo_ratings[home_team] = new_home_ranking
			new_away_ranking = 1500 + (elo_ratings[away_team] - 1500) * mean_reversion
			elo_ratings[away_team] = new_away_ranking



		home_score = row['Home Score']
		away_score = row['Away Score']
		home_ranking = elo_ratings[home_team]
		away_ranking = elo_ratings[away_team]

		# Estimate odds for games
		#Pr(A) = 1 / (10^(-ELODIFF/400) + 1)
		prob_home = 1 / (10**((away_ranking-home_ranking)/400)+1)
		prob_away = 1 / (10**((home_ranking-away_ranking)/400)+1)
		afl_results.ix[row_index,'home_prob'] = prob_home
		afl_results.ix[row_index,'away_prob'] = prob_away
		afl_results.ix[row_index,'home_odds_estimate'] = 1 / prob_home
		afl_results.ix[row_index,'away_odds_estimate'] = 1 / prob_away
		if home_score > away_score:
			afl_results.ix[row_index,'home_result'] = 1
			afl_results.ix[row_index,'away_result'] = 0
		elif home_score == away_score:
			afl_results.ix[row_index,'home_result'] = 0.5
			afl_results.ix[row_index,'away_result'] = 0.5
		else:
			afl_results.ix[row_index,'home_result'] = 0
			afl_results.ix[row_index,'away_result'] = 1




		point_differential = home_score - away_score
		if home_score > away_score:
			winners_rank = home_ranking
			losers_rank = away_ranking
		else:
			winners_rank = away_ranking
			losers_rank = home_ranking
		
		#home team calculation
		old_rank = home_ranking
		opponent_rank = away_ranking
		home_field_advantage = home_field
		if home_score > away_score:
			actual_result = 1
		elif home_score == away_score:
			actual_result = 0.5
		else:
			actual_result = 0
		new_rank = elo_calculation(k,point_differential,old_rank,opponent_rank, winners_rank,losers_rank,actual_result, home_field_advantage,margin_beta)
		elo_ratings[home_team] = new_rank
		elo_historical[home_team]['Date'].append(row['Date'])
		elo_historical[home_team]['Elo'].append(new_rank)
		
		#away team calculation
		old_rank = away_ranking
		opponent_rank = home_ranking
		home_field_advantage = - home_field
		if away_score > home_score:
			actual_result = 1
		elif away_score == home_score:
			actual_result = 0.5
		else:
			actual_result = 0
		new_rank = elo_calculation(k,point_differential,old_rank,opponent_rank, winners_rank,losers_rank,actual_result, home_field_advantage,margin_beta)
		elo_ratings[away_team] = new_rank
		elo_historical[away_team]['Date'].append(row['Date'])
		elo_historical[away_team]['Elo'].append(new_rank)

	return afl_results, elo_ratings, elo_historical



def test_parameters():
	# for j in range(10,30,2):
	# 	for k in range(0,200,10):
	# 		afl_results = afl_elo_timeseries(afl_results,j,k)
	# 		error = ((afl_results['home_result'] - afl_results['home_prob'])**2).sum()
	# 		print(j,k,error)
	for j in np.arange(0.0001,0.002,0.0001):
		results = afl_elo_timeseries(afl_results,16,110,0.7,12)
		error = ((results[0]['home_result'] - results[0]['home_prob'])**2).sum()
		print(j,error)




#test_parameters()





def show_chart(data_frame,k,home_field,mean_reversion,margin_beta,teams=False):
	import matplotlib.pyplot as plt
	data_frame, elo_ratings, elo_historical = afl_elo_timeseries(data_frame,k,home_field,mean_reversion,margin_beta)
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


show_chart(afl_results,16,110,0.70,12.2,['Sydney','Hawthorn','West Coast','Fremantle','Geelong','Port Adelaide'])

def bet_test(data_frame,k,home_field,mean_reversion,margin_beta):
	results = afl_elo_timeseries(data_frame,k,home_field,mean_reversion,margin_beta)
	total_bet = 0
	total_win = 0
	
	test_bet = 0
	test_win = 0 

	for row_index, row in results[0].iterrows():
		if row['Away Odds'] > row['away_odds_estimate']:
			b = row['Away Odds'] - 1
			p = 1 / row['away_odds_estimate']
			bet = (p *  ( b + 1 ) - 1) / b
			total_bet += bet
			if row['Away Score'] > row['Home Score']:
				total_win += row['Away Odds'] * bet
		elif row['Home Odds'] > row['home_odds_estimate']:
			b = row['Home Odds'] - 1
			p = 1 / row['home_odds_estimate']
			bet = (p *  ( b + 1 ) - 1) / b
			total_bet += bet
			if row['Home Score'] > row['Away Score']:
				total_win += row['Home Odds'] * bet

		if row['Away Odds'] > row['away_odds_estimate']:
			test_bet += 1
			if row['Home Score'] > row['Away Score']:
				test_win += row['Home Odds']
		elif row['Home Odds'] > row['home_odds_estimate']:
			test_bet += 1
			if row['Away Score'] > row['Home Score']:
				test_win += row['Away Odds']


	print(total_bet,total_win,total_win/total_bet)
	print(test_win/test_bet)
	



# bet_test(afl_results,16,110,0.70,12.2)









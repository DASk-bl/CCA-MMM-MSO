1) The data that are connected to Funnel.io is exported via multipe Big Query exports where they are processed and combined into a single BQ table with the necessary columns.

2) All the data for the CCA are stored in Big Query Table `funnel-data.funnel_export_n_3857424256_namer_cca_global_partners_contains.Robin_Oppiste_Rule1` 

3) The problem here would be the locations that has 0 historical spend aswell present that increases the time while querying the data for in the UI. 

4) So a new table was created which has only the location with some historical spend `funnel-data.funnel_export_n_3857424256_namer_cca_global_partners_contains.Latest_table`

5) In the Repo you can find Multiple different files.
	
	(i) explore_model --> This contain the codes of data required to plot all the graphs on the Explore page of the UI.
	
	(ii) predict_model --> This containts the code for for the MMM part of the tool where in it gives the response curves for the channel that are 	involved in the MMM part and their respective accuracy in terms of SMAPE. Apart from that this code provrs the data to plot the 100% stacked area 	chart that is plotted in the Predict page of the UI.
	
	(iii) optimise_model --> This contains the code for the MSO functionality of the tool, based on the input investment, the number of months and the 	min-max bounds spcification for each channel the model gives the proper optimisation plan based on all the factors. 


Notes: 
-- A location needs to have more than 12 months of data inorder for the code to work because thats when the seasonality part gets computed.
-- The entire model is proposed to run and be retrained for every 3 months inorder to get the best outcome of the current spending.
-- At the current state the is huge amount of spend on search channel than the others so the output of our tool in any aspect will favour Search.
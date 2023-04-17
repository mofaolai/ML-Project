# ML-Project

I've added code in 'combine_data.ipynb'
It reads in all the data files and combines them into one (full_data), with an added column called usage.  
if you put your data into a folder called dublinbikes in your ML-PROJECT folder and create a data folder, it should run for you.


usage is just the diff between AVAILABLE_BIKES - it's wrong for each of the first rows of  the stations, so if one of you can fix it?
might need to sort the data by station name first?

for q1 - once you have a value for usage, I think that's what he wants us to plot - maybe absolute value? summed over the week?


A: Really like your clustering thing; can you pick another station from each cluster - there's a station at the mater hospital too.   if you add the names to station_names in the code, it'll automatically add them to the data file

M: couldn't read your file - jupyter didn't recognise it.


anyway, I'm going to look at modelling based on usage, so if we decide to change the definition, we just need to put it in the same column.
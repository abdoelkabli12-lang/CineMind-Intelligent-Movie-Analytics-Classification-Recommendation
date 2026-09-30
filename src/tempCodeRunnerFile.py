avg_budget_per_year = mycol.aggregate([{
#   "$group" : {"_id" : None, "average"}
# }])
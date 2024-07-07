import pandas as pd

user_directories = [
    "Dashboard_user_id_24026_3",
    "Dashboard_user_id_24491_0",
    "Dashboard_user_id_35133_0",
    "Dashboard_user_id_38058_0",
    "Dashboard_user_id_49381_0"
]
runningPeripheralInputDfs = []

for user in user_directories:
    peripheralInputs = pd.read_json(f"./results/{user}.json", orient='index')
    runningPeripheralInputDfs.append(peripheralInputs)

# concatenate along rows
concatenatedPeripheralInputs = pd.concat(runningPeripheralInputDfs, axis=0)
concatenatedPeripheralInputs.to_json('./results/ALL.json', orient='index')
print("Combined peripheral inputs stored in:")
print("./results/ALL.json")


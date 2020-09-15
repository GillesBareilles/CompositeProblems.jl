function checkdownload_ionosphere()
    if !isfile("instances/ionosphere.data")
        location = "instances"
        println("Downloading ionosphere dataset to: ", location, "...")

        run(`wget https://archive.ics.uci.edu/ml/machine-learning-databases/ionosphere/ionosphere.data`)

        !isdir(location) && mkdir(location)
        mv("ionosphere.data", joinpath(location, "ionosphere.data"))

        println("All done.")
    end
    return
end

function checkdownload_ionosphere()
    if !isfile(joinpath(instances_dir, "ionosphere.data"))
                println("Downloading ionosphere dataset to: ", pwd(), "...")

        run(`wget https://archive.ics.uci.edu/ml/machine-learning-databases/ionosphere/ionosphere.data`)

        mv("ionosphere.data", joinpath(instances_dir, "ionosphere.data"))
        println("Moving ionosphere dataset to: ", joinpath(instances_dir, "ionosphere.data"), "...")

        println("All done.")
    end
    return
end

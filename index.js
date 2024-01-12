const WebSocket = require('ws');
const { PythonShell } = require("python-shell");


const wss = new WebSocket.Server({ port: 5001 });

async function run(ws, name, options, id) {
    try {
        let result = await runPythonScript(ws, "./Functions/" + name + ".py", options, id);
        console.log("Python script output:", result);
        // Process and send result here
    } catch (err) {
        console.error('Error running Python script:', err);
    }
}

// Function to run Python script
async function runPythonScript(ws, scriptPath, options, id) {
    // Move the declaration inside the function
    
    return new Promise((resolve, reject) => {
        let individualAlgorithmExecutionResults = [];
        let shell = new PythonShell(scriptPath, options);
        let output = [];

        shell.on('message', function (message) {
            // console.log("Here", message)
        });

        shell.on('stderr', function (stderr) {
            let parts = stderr.split(',');
            let epochPart = parts.find(part => part.includes('Epoch'));
            let currentBestPart = parts.find(part => part.includes('Current best'));
            if (epochPart) {
                let epochValue = epochPart.split(':')[1].trim();
                let currentBestValue = currentBestPart.split(':')[1].trim();

                individualAlgorithmExecutionResults.push({
                    iterationNumber: parseInt(epochValue),
                    iterationFitnessScore: parseFloat(currentBestValue)
                });
            }
        });

        shell.end(function (err) {
            if (!err) {
                ws.send(JSON.stringify({
                    executionId: id,
                    results: individualAlgorithmExecutionResults
                }));
                resolve({message: "algorithm execution results sent successfully"});
            } else {
                reject(err);
            }
        });
    });
}


wss.on('connection', function connection(ws) {
    ws.on('message', async function incoming(message) {
        console.log('Client is connected');
        const messageString = message.toString();
        try {
            const messageJSON = JSON.parse(messageString);

            for (const [index, msg] of messageJSON.entries()) {
                console.log(msg.algorithmCode);
                let algorithmName = msg.algorithmCode;
                //YORUM
                switch (algorithmName) {
                    case "SA":
                        console.log(msg)
                        let id = msg.executionId
                        let options = {
                            args: [
                                msg.nVars,
                                msg.lb,
                                msg.ub,
                                msg.initialTemperature,
                                msg.coolDownFactor,
                                msg.temperatureDecreaseType,
                                msg.populationSize,
                                msg.epoch,
                                msg.selectedBenchmarkFunction,
                                msg.minmax
                            ]
                        };
                        await run(ws, "SA", options, id)
                        break;
                    case "PSO":
                        console.log(msg)
                        let idPso = msg.executionId
                        let optionsPSO = {
                            args: [
                                msg.lb,
                                msg.ub,
                                msg.populationSize,
                                msg.numberOfGenerations,
                                msg.c1,
                                msg.c2,
                                msg.w,
                                msg.selectedBenchmarkFunction,
                                msg.minmax,
                                msg.nVars
                            ]
                        }
                        await run(ws, "PSO", optionsPSO, idPso)
                        break;
                    case "MGO":
                        console.log(msg)
                        let idMgo = msg.executionId
                        console.log(msg)
                        let optionsMgo = {
                            args: [
                                msg.nVars,
                                msg.lb,
                                msg.ub,
                                msg.minmax,
                                msg.selectedBenchmarkFunction,
                                msg.epoch,
                                msg.populationSize
                            ]
                        }
                        await run(ws, "MGO", optionsMgo, idMgo)
                        break;
                    case "MPA":
                        let idMPA = msg.executionId
                        console.log(msg)
                        let optionsMpa = {
                            args: [
                                msg.nVars,
                                msg.lb,
                                msg.ub,
                                msg.populationSize,
                                msg.selectedBenchmarkFunction,
                                msg.minmax,
                                msg.epoch,
                            ]
                        }
                        await run(ws, "MPA", optionsMpa, idMPA)
                        break;
                    case "HGSO":
                        console.log(msg)
                        const idHgso = msg.executionId
                        let optionsHgso = {
                            args: [
                                msg.nVars,
                                msg.lb,
                                msg.ub,
                                msg.minmax,
                                msg.selectedBenchmarkFunction,
                                msg.epoch,
                                msg.populationSize,
                                msg.nClusters
                            ]
                        }
                        await run(ws,"HGSO",optionsHgso,idHgso)
                        break
                    case "HSA":
                        let hsId = msg.executionId
                        console.log(msg)
                        let optionsHs = {
                            args: [
                                msg.lb,
                                msg.ub,
                                msg.PAR,
                                msg.nVars,
                                msg.minmax,
                                msg.cR,
                                msg.selectedBenchmarkFunction,
                                msg.epoch,
                                msg.popSize
                            ]
                        }
                        await run(ws,"HS",optionsHs,hsId)
                        break
                        case "GWO":
                            console.log(msg)    
                            let idGwo = msg.executionId
                            let optionsGWO = {
                                args: [
                                    msg.populationSize,
                                    msg.lb,
                                    msg.ub,
                                    msg.nVars,
                                    msg.minmax,
                                    msg.numberOfGenerations,
                                    msg.decreaseFrom,
                                    msg.selectedBenchmarkFunction,
                                ]
                            }
                            await run(ws, "GWO", optionsGWO, idGwo)
                            break;
                        case "EO":
                            console.log(msg)
                        //     // PSO case logic here
                            let idEo = msg.executionId
                            let optionsEO = {
                                args: [
                                    msg.nVars,
                                    msg.lb,
                                    msg.ub,
                                    msg.selectedBenchmarkFunction,
                                    msg.epoch,
                                    msg.populationSize,
                                    msg.minmax
                                ]
                            }
                            await run(ws, "EO", optionsEO, idEo)
                            break;
                        case "AGTO":
                            console.log(msg)
                            let idAgto = msg.executionId
                            let optionsAGTO = {
                                args: [
                                    msg.nVars,
                                    msg.lb,
                                    msg.ub,
                                    msg.selectedBenchmarkFunction,
                                    msg.minmax,
                                    msg.epoch,
                                    msg.populationSize,
                                    msg.p1,
                                    msg.p2,
                                    msg.beta
                                ]
                            }
                            await run(ws, "AGTO", optionsAGTO, idAgto)
                            break;
                            case "AOA":
                                console.log(msg)
                                let idAoa = msg.executionId
                                let optionsAOA = {
                                    args: [
                                        msg.nVars,
                                        msg.lb,
                                        msg.ub,
                                        msg.minmax,
                                        msg.selectedBenchmarkFunction,
                                        msg.epoch,
                                        msg.populationSize,
                                        msg.alpha,
                                        msg.miu,
                                        msg.moaMin,
                                        msg.moaMax
                                    ]
                                }
                                await run(ws, "AOA", optionsAOA, idAoa)
                                break;
                            case "AVO":
                                console.log(msg)
                                let idAvo = msg.executionId
                                let optionsAVO = {
                                    args: [
                                        msg.lb,
                                        msg.ub,
                                        msg.nVars,
                                        msg.selectedBenchmarkFunction,
                                        msg.minmax,
                                        msg.epoch,
                                        msg.populationSize,
                                        msg.p1,
                                        msg.p2,
                                        msg.p3,
                                        msg.alpha,
                                        msg.gama
                                    ]
                                }
                                await run(ws, "AVOA", optionsAVO, idAvo)
                                break;
                            case "BMO":
                                console.log(msg)
                                let idBmo = msg.executionId
                                let optionsBMO = {
                                    args: [
                                        msg.nVars,
                                        msg.lb,
                                        msg.ub,
                                        msg.name,
                                        msg.minmax,
                                        msg.selectedBenchmarkFunction,
                                        msg.epoch,
                                        msg.popSize,
                                        msg.pl,
                                    ]
                                }
                                await run(ws, "BMO", optionsBMO, idBmo)
                                break;
                            case "BBO":
                                console.log(msg)
                                let idBbo = msg.executionId
                                let optionsBBO = {
                                    args: [
                                        msg.nVars,
                                        msg.lb,
                                        msg.ub,
                                        msg.name,
                                        msg.minmax,
                                        msg.selectedBenchmarkFunction,
                                        msg.epoch,
                                        msg.popSize,
                                        msg.pM,
                                        msg.nElites,
                                    ]
                                }
                                await run(ws, "BBO", optionsBBO, idBbo)
                                break;
                            case "BRBO":
                                console.log(msg)
                                let idBboa = msg.executionId
                                let optionsBBOA = {
                                    args: [
                                        msg.nVars,
                                        msg.lb,
                                        msg.ub,
                                        msg.name,
                                        msg.minmax,
                                        msg.selectedBenchmarkFunction,
                                        msg.epoch,
                                        msg.popSize,
                                    ]
                                }
                                await run(ws, "BBOA", optionsBBOA, idBboa)
                                break;
                            case "CRO":
                                console.log(msg)
                                let idCro = msg.executionId
                                let optionsCRO = {
                                    args: [
                                        msg.nVars,
                                        msg.lb,
                                        msg.ub,
                                        msg.name,
                                        msg.minmax,
                                        msg.selectedBenchmarkFunction,
                                        msg.epoch,
                                        msg.popSize,
                                        msg.po,
                                        msg.Fb,
                                        msg.Fa,
                                        msg.Fd,
                                        msg.Pd,
                                        msg.GCR,
                                        msg.gamma_min,
                                        msg.gamma_max,
                                        msg.nTrials,
                                    ]
                                }
                                await run(ws, "CRO", optionsCRO, idCro)
                                break;
                            case "DE":
                                console.log(msg)
                                let idDe = msg.executionId
                                let optionsDE = {
                                    args: [
                                        msg.nVars,
                                        msg.lb,
                                        msg.ub,
                                        msg.name,
                                        msg.minmax,
                                        msg.selectedBenchmarkFunction,
                                        msg.epoch,
                                        msg.popSize,
                                        msg.strategy,
                                        msg.wf,
                                        msg.cr,
                                    ]
                                }
                                await run(ws, "DE", optionsDE, idDe)
                                break;
                            case "EOA":
                                console.log(msg)
                                let idEoa = msg.executionId
                                let optionsEOA = {
                                    args: [
                                        msg.nVars,
                                        msg.lb,
                                        msg.ub,
                                        msg.name,
                                        msg.minmax,
                                        msg.selectedBenchmarkFunction,
                                        msg.epoch,
                                        msg.popSize,
                                        msg.pC,
                                        msg.pM,
                                        msg.nBest,
                                        msg.alpha,
                                        msg.beta,
                                        msg.gamma,
                                    ]
                                }
                                await run(ws, "EOA", optionsEOA, idEoa)
                                break;
                            case "EP":
                                console.log(msg)
                                let idEp = msg.executionId
                                let optionsEP = {
                                    args: [
                                        msg.nVars,
                                        msg.lb,
                                        msg.ub,
                                        msg.name,
                                        msg.minmax,
                                        msg.selectedBenchmarkFunction,
                                        msg.epoch,
                                        msg.popSize,
                                        msg.boutSize,
                                    ]
                                }
                                await run(ws, "EP", optionsEP, idEp)
                                break;
                            case "ES":
                                console.log(msg)
                                let idEs = msg.executionId
                                let optionsES = {
                                    args: [
                                        msg.nVars,
                                        msg.lb,
                                        msg.ub,
                                        msg.name,
                                        msg.minmax,
                                        msg.selectedBenchmarkFunction,
                                        msg.epoch,
                                        msg.popSize,
                                        msg.lambda,
                                    ]
                                }
                                await run(ws, "ES", optionsES, idEs)
                                break;
                            case "FPA":
                                console.log(msg)
                                let idFpa = msg.executionId
                                let optionsFPA = {
                                    args: [
                                        msg.nVars,
                                        msg.lb,
                                        msg.ub,
                                        msg.name,
                                        msg.minmax,
                                        msg.selectedBenchmarkFunction,
                                        msg.epoch,
                                        msg.popSize,
                                        msg.pS,
                                        msg.levyMultiplier,
                                    ]
                                }
                                await run(ws, "FPA", optionsFPA, idFpa)
                                break;
                            case "IWO":
                                console.log(msg)
                                let idIwo = msg.executionId
                                let optionsIWO = {
                                    args: [
                                        msg.nVars,
                                        msg.lb,
                                        msg.ub,
                                        msg.name,
                                        msg.minmax,
                                        msg.selectedBenchmarkFunction,
                                        msg.epoch,
                                        msg.popSize,
                                        msg.seedMin,
                                        msg.seedMax,
                                        msg.exponent,
                                        msg.sigmaStart,
                                        msg.sigmaEnd,
                                    ]
                                }
                                await run(ws, "IWO", optionsIWO, idIwo)
                                break;
                            case "SOS":
                                console.log(msg)
                                let idSos = msg.executionId
                                let optionsSOS = {
                                    args: [
                                        msg.nVars,
                                        msg.lb,
                                        msg.ub,
                                        msg.name,
                                        msg.minmax,
                                        msg.selectedBenchmarkFunction,
                                        msg.epoch,
                                        msg.popSize,
                                    ]
                                }
                                await run(ws, "SOS", optionsSOS, idSos)
                                break;
                            case "TPO":
                                console.log(msg)
                                let idTpo = msg.executionId
                                let optionsTPO = {
                                    args: [
                                        msg.nVars,
                                        msg.lb,
                                        msg.ub,
                                        msg.name,
                                        msg.minmax,
                                        msg.selectedBenchmarkFunction,
                                        msg.epoch,
                                        msg.popSize,
                                        msg.alpha,
                                        msg.beta,
                                        msg.theta
                                    ]
                                }
                                await run(ws, "TPO", optionsTPO, idTpo)
                                break;
                            case "TSA":
                                console.log(msg)
                                let idTsa = msg.executionId
                                let optionsTSA = {
                                    args: [
                                        msg.nVars,
                                        msg.lb,
                                        msg.ub,
                                        msg.name,
                                        msg.minmax,
                                        msg.selectedBenchmarkFunction,
                                        msg.epoch,
                                        msg.popSize,
                                    ]
                                }
                                await run(ws, "TSA", optionsTSA, idTsa)
                                break;
                            case "VCS":
                                console.log(msg)
                                let idVcs = msg.executionId
                                let optionsVCS={
                                    args: [
                                        msg.nVars,
                                        msg.lb,
                                        msg.ub,
                                        msg.name,
                                        msg.minmax,
                                        msg.selectedBenchmarkFunction,
                                        msg.epoch,
                                        msg.popSize,
                                        msg.lamda,
                                        msg.sigma,
                                    ]
                                }
                                await run(ws, "VCS", optionsVCS, idVcs)
                                break;
                            case "WHO":
                                console.log(msg)
                                let idWho = msg.executionId
                                let optionsWHO={
                                    args: [
                                        msg.nVars,
                                        msg.lb,
                                        msg.ub,
                                        msg.name,
                                        msg.minmax,
                                        msg.selectedBenchmarkFunction,
                                        msg.epoch,
                                        msg.popSize,
                                        msg.nExploreStep,
                                        msg.nExploitStep,
                                        msg.eta,
                                        msg.pHi,
                                        msg.localAlpha,
                                        msg.localBeta,
                                        msg.globalAlpha,
                                        msg.globalBeta,
                                        msg.deltaW,
                                        msg.deltaC,
                                    ]
                                }
                                await run(ws, "WHO", optionsWHO, idWho)
                                break;
                            case "GA":
                                console.log(msg)
                                if(msg.selectionType == 'tournament'){
                                    let idGa =  msg.executionId
                                        let optionsGA={
                                            args: [
                                                msg.nVars,
                                                msg.populationSize,
                                                msg.lb,
                                                msg.ub,
                                                msg.numberOfGenerations,
                                                msg.mutationProbability,
                                                msg.crossoverProbability,
                                                msg.crossoverType,
                                                msg.selectionType,
                                                msg.mutationMultiPoints,
                                                msg.mutationType,
                                                msg.kWay,
                                                msg.selectedBenchmarkFunction
                                            ]
                                        }
                                        await run(ws, "GA", optionsGA, idGa)
                                        break;
                                }
                                else{
                                    let idGa2 =  msg.executionId
                                        let optionsGA2={
                                            args: [
                                                msg.nVars,
                                                msg.populationSize,
                                                msg.lb,
                                                msg.ub,
                                                msg.numberOfGenerations,
                                                msg.mutationProbability,
                                                msg.crossoverProbability,
                                                msg.crossoverType,
                                                msg.selectionType,
                                                msg.mutationMultiPoints,
                                                msg.mutationType,
                                                msg.selectedBenchmarkFunction
                                            ]
                                        }
                                        await run(ws, "GA2", optionsGA2, idGa2)
                                        break;
                                }
                                case "MA":
                                    console.log(msg)
                                    let idMA = msg.executionId
                                    let optionsMA = {
                                        args: [
                                            msg.nVars,
                                            msg.lb,
                                            msg.ub,
                                            msg.name,
                                            msg.minmax,
                                            msg.selectedBenchmarkFunction,
                                            msg.epoch,
                                            msg.popSize,
                                            msg.pc,
                                            msg.pm,
                                            msg.pLocal,
                                            msg.maxLocalGenes,
                                            msg.bitsPerParam
                                        ]
                                    }
                                    await run(ws, "MA", optionsMA, idMA)
                                    break;
                                case "SBBO":
                                    console.log(msg);
                                    let idSbo = msg.executionId
                                    let optionsSbo = {
                                        args: [
                                            msg.nVars,
                                            msg.lb,
                                            msg.ub,
                                            msg.name,
                                            msg.minmax,
                                            msg.selectedBenchmarkFunction,
                                            msg.epoch,
                                            msg.popSize,
                                            msg.alpha,
                                            msg.pM,
                                            msg.psw,
                                        ]
                                    }
                                    await run(ws, "SBO", optionsSbo, idSbo)
                                    break;
                                case "SHADE":
                                    console.log(msg)
                                    let idShade = msg.executionId
                                    let optionsShade = {
                                        args: [
                                            msg.nVars,
                                            msg.lb,
                                            msg.ub,
                                            msg.name,
                                            msg.minmax,
                                            msg.selectedBenchmarkFunction,
                                            msg.epoch,
                                            msg.popSize,
                                            msg.miuF,
                                            msg.miuCR
                                        ]
                                    }
                                    await run(ws, "SHADE", optionsShade, idShade)
                                    break;
                                case "SMA":
                                    console.log(msg);
                                    let idSMO = msg.executionId
                                    let optionsSMO = {
                                        args: [
                                            msg.nVars,
                                            msg.lb,
                                            msg.ub,
                                            msg.name,
                                            msg.minmax,
                                            msg.selectedBenchmarkFunction,
                                            msg.epoch,
                                            msg.popSize,
                                            msg.pT
                                        ]
                                    }
                                    await run(ws, "SMA", optionsSMO, idSMO)
                                    break;
                                case "SOA":
                                    console.log(msg)
                                    let idSoa = msg.executionId
                                    let optionsoa = {
                                        args: [
                                            msg.nVars,
                                            msg.lb,
                                            msg.ub,
                                            msg.name,
                                            msg.minmax,
                                            msg.selectedBenchmarkFunction,
                                            msg.epoch,
                                            msg.popSize,
                                            msg.fc
                                        ]
                                    }
                                    await run(ws, "SOA", optionsoa, idSoa)
                                    break;
                } // run(ws, "DOSYANIN ADI", optionsXXX, idXXX)
            }

        } catch (error) {
            console.error('Error parsing JSON:', error);
        }
    });

    ws.on('close', () => {
        console.log('Client disconnected');
    });
});

console.log('WebSocket server started on ws://localhost:5001');

// case "MPA":
//                         let idMPA = msg.executionId
//                         console.log(msg)
//                         let optionsMpa = {
//                             args : [
//                                 msg.nVars,
//                                 msg.minmax,
//                                 msg.benchmarkFunction,
//                                 msg.epoch,
//                                 msg.populationSize
//                             ]
//                         }
//                         run(ws,"MPA",optionsMpa,idMPA)
//                         break;
# Copyright 2017 Eric S. Tellez
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

module Scores

export accuracy, precision_recall, precision, recall, f1

function recall(gold, predict; weight=:macro)
    if weight == :macro
        mean(x -> x.second[2], precision_recall(gold, predict))
    else  ## :micro
        mean(x -> x.second[2] * x.second[3]/length(gold), precision_recall(gold, predict))
    end
end

function precision(gold, predict; weight=:macro)
    if weight == :macro
        mean(x -> x.second[1], precision_recall(gold, predict))
    else  ## :micro
        mean(x -> x.second[1] * x.second[3]/length(gold), precision_recall(gold, predict))
    end
end

function f1(gold, predict; weight=:macro)
    if weight == :macro
        mean(x -> 2 * x.second[1] * x.second[2] / (x.second[1] + x.second[2]), precision_recall(gold, predict))
    else  ## :micro
        mean(x -> 2 * x.second[1] * x.second[2] / (x.second[1] + x.second[2]) * x.second[3]/length(gold), precision_recall(gold, predict))
    end
end

function precision_recall(gold, predicted)
    labels = unique(gold)
    M = Dict{typeof(labels[1]), Tuple}()
    for label in labels
        lgold = label .== gold
        lpred = label .== predicted

        tp = 0
        tn = 0
        fn = 0
        fp = 0
        for i in 1:length(lgold)
            if lgold[i] == lpred[i]
                if lgold[i]
                    tp += 1
                else
                    tn += 1
                end
            else
                if lgold[i]
                    fn += 1
                else
                    fp += 1
                end
            end
        end

        M[label] = (tp / (tp + fp), tp/(tp + fn), sum(lgold) |> Int)  # precision, recall, class-population
    end

    M
end

function accuracy(gold, predicted)
    #  mean(gold .== predicted)
    c = 0
    for i in 1:length(gold)
        c += (gold[i] == predicted[i])
    end

    c / length(gold)
end

end

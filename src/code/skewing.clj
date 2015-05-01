(ns skewing.core
  (:require [skewing.core :as core]))
;assume we have instances
(defn skew
  "returns a seq of instances and their weights"
  [instances]
  (let [preferred]
    (for [inst instances]
      [inst ])))

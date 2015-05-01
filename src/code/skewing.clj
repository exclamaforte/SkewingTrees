(ns skewing.skew
  (:require [skewing.core :as core]))
;assume we have instances
(use 'print.foo)
(defn skew
  "returns a seq of instances and their weights"
  [instances c]
  (let [attr-seq (core/attribute-seq instances)
        preferred (into {} (for [x attr-seq]
                             [x (rand-nth (core/attribute-vals x))]))]
    (into {} (for [inst instances]
               [inst
                (reduce * 1 (for [attr attr-seq]
                              (if (= (. inst stringValue attr)
                                     (get preferred attr))
                                c
                                1)))]))))

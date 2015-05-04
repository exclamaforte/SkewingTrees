(ns skew.trees-testb
  (:require [clojure.test.check :as tc]
            [clojure.test.check.generators :as gen]
            [clojure.test.check.properties :as prop]
            [clojure.test.check.clojure-test :as ct]
            [code.decision-trees :as dt]
            [clojure.set]))

(conditional-entropy (core/nth-attribute (core/nth-instance core/test-instances 1) 0) core/test-instances)

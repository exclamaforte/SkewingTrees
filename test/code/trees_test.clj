(ns skew.trees-test
  (:require [clojure.test.check :as tc]
            [clojure.test.check.generators :as gen]
            [clojure.test.check.properties :as prop]
            [clojure.test.check.clojure-test :as ct]
            [code.decision-trees :as dt]
            [clojure.set]))

(ct/defspec dt-conserve-elements
  50
  (prop/for-all [])) ;write test that sees if the data is conserved within a tree
(conditional-entropy (core/nth-attribute (core/nth-instance core/test-instances 1) 0) core/test-instances)
(build-decision-tree core/test-instances)
(rf-classify (first core/test-instances) (build-random-forest core/test-instances 2))

(doseq [tree (build-random-forest core/test-instances 10)]
  (print tree)
  (print "~!@#$")
  (print (dt-classify (first core/test-instances) tree)))

---
title: Home
layout: home
---

# Rachel Docs

chasing questions - psychology and ai

---

## 최신 포스트

{% assign recent_posts = site.posts | sort: "date" | reverse | slice: 0, 5 %}

{% for post in recent_posts %}
### [{{ post.title }}]({{ post.url }})
**{{ post.date | date: "%Y-%m-%d" }}** | 카테고리: {{ post.categories | join: ", " }}

{% if post.subtitle %}
*{{ post.subtitle }}*
{% endif %}

{% if post.excerpt %}
{{ post.excerpt | strip_html | truncatewords: 30 }}
{% endif %}

---
{% endfor %}

[모든 포스트 보기 →]({{ site.baseurl }}/categories.html)
